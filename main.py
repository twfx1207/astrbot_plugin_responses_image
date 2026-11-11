"""Expose isolated image generation and explicit character references."""

import asyncio
import copy
import json
import re
import time
from uuid import uuid4

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.provider.sources.openai_responses_source import (
    ProviderOpenAIResponses,
)
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
from astrbot.core.star.filter.command import GreedyStr
from mcp.types import CallToolResult

from . import background, delivery, generation, request_context
from .provider import (
    ImagesAPIClient,
    ResponsesImageClient,
)
from .quota import DrawingQuota, QuotaExceeded
from .references import CharacterLibrary, collect_event_images


class ResponsesImagePlugin(Star):
    """Coordinate isolated image generation and main-agent image feedback."""

    def __init__(self, context: Context, config: dict | None = None) -> None:
        """Initialize the plugin without changing any existing provider.

        Args:
            context: AstrBot plugin context.
            config: Optional agent models, image model and reference search settings.
        """
        super().__init__(context)
        self.config = config if config is not None else {}
        self.library: CharacterLibrary | None = None
        self._jobs: dict[str, dict] = {}
        self._closing = False
        self.quota: DrawingQuota | None = None
        self._queue_changed = asyncio.Condition()

    async def initialize(self) -> None:
        """Open the character library without registering or replacing providers."""
        self._closing = False
        self.logger.debug(
            "Background image completion backend: %s (native_import=%s, legacy_import=%s)",
            "native"
            if background.wake_main_agent is not None
            else (
                "legacy"
                if background.run_legacy_completion is not None
                else "unavailable"
            ),
            background.native_wakeup_import_error,
            background.legacy_wakeup_import_error,
        )
        if self.library is None:
            self.library = CharacterLibrary(
                StarTools.get_data_dir("astrbot_plugin_responses_image")
            )
        if self.quota is None:
            self.quota = DrawingQuota(self.library.data_dir / "quota.json", self.config)
        self.context.register_web_api(
            "/responses_image/queue",
            self.queue_status,
            ["GET"],
            "Inspect queued and running image tasks",
        )

    @filter.on_llm_request()
    async def prepare_image_request(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """Attach request-local drawing context and delivery tools.

        Args:
            event: Current chat or background completion event.
            req: Main agent request whose tool set and instructions are updated.
        """
        await request_context.prepare_image_request(self, event, req)

    @filter.llm_tool(name="responses_draw_image")
    async def draw_image(
        self,
        event: AstrMessageEvent,
        operation: str,
        character_names: list[str],
        reference_ids: list[str],
        context_ids: list[str] | None = None,
        search_references: bool = True,
    ):
        """Create or edit an image in an isolated request using the current user's words.

        Never pass a rewritten prompt or arbitrary historical images. Use this only for an
        explicit image request. Resolve clear pronouns from selected recent text.
        Existing named characters require separate character_names entries;
        Optional search checks character references; if unavailable, new images
        use the character names alone. Edits always need a selected target image.
        Decide whether reference research is necessary before calling; no-search
        requests must set search_references=false even when a character is named.
        Successful images return to the main agent for review and delivery.

        Args:
            operation(string): new for a new composition, or edit for an explicitly selected image.
            character_names(array[string]): Up to four existing character names or aliases; empty for original subjects or edits.
            reference_ids(array[string]): Explicit current reference IDs, or latest generated edit IDs when editing that image; empty for text-only images or registered characters.
            context_ids(array[string]): Recent text IDs needed to resolve the intended characters; empty when names are in the current message.
            search_references(boolean): Whether this request needs external character reference research. Set false for original or text-specified art, or when the user does not want reference lookup. Current images, edits and registered references do not require searching. True only enables missing-character research when the panel permits it.

        Returns:
            A multimodal tool result on success, or an explanatory error string.
        """
        if (
            (
                (
                    background.AgentWakeupEvent is not None
                    and isinstance(event, background.AgentWakeupEvent)
                )
                or (
                    background.LegacyCompletionEvent is not None
                    and isinstance(event, background.LegacyCompletionEvent)
                )
            )
            and event.source == "plugin"
            and (event.source_id or "").startswith("responses-image-")
        ):
            job = event.get_extra("responses_image_completed_job")
            if (
                not job
                or job["stop"].is_set()
                or job["umo"] != event.unified_msg_origin
            ):
                return "后台结果已经失效；本轮不会重新发起生图。"
            if job["claimed"].is_set():
                return "本任务结果已取回，请使用已有结果，不要重复获取或生成。"
            result = job["result"]
            job["result"] = None
            job["claimed"].set()
            if isinstance(result, CallToolResult):
                event.set_extra("responses_image_ready", True)
                event.set_extra("responses_image_requester", job["submitter"][1])
            return result
        if type(search_references) is not bool:
            return (
                "请先判断是否需要搜索参考图，search_references 只能为 true 或 false。"
            )
        if not isinstance(operation, str) or operation not in {"new", "edit"}:
            return "生图操作只能是 new 或 edit。"
        if not isinstance(character_names, list) or not isinstance(reference_ids, list):
            return "角色名和参考图参数格式不正确。"
        if len(character_names) > 4 or any(
            not isinstance(name, str) or not name.strip() or len(name) > 200
            for name in character_names
        ):
            return (
                "一次最多绘制四个角色，请将每个角色名分别填写，单个名字不超过 200 字。"
            )
        if len(reference_ids) > 4 or any(
            not isinstance(key, str) for key in reference_ids
        ):
            return "一次最多选择四张当前可用的参考图或编辑目标。"
        character_names = [name.strip() for name in character_names]
        context_ids = [] if context_ids is None else context_ids
        if (
            not isinstance(context_ids, list)
            or len(context_ids) > 8
            or any(not isinstance(key, str) for key in context_ids)
        ):
            return "上下文选择参数格式不正确，请选择本次提供的文字片段编号。"
        context_candidates = event.get_extra("responses_image_contexts", {})
        if any(key not in context_candidates for key in context_ids):
            return "上下文编号不属于本次消息，请使用当前可用的对话片段解析角色。"
        selected_context = [
            context_candidates[key]["text"] for key in dict.fromkeys(context_ids)
        ]
        references = collect_event_images(event)
        if operation == "edit":
            references = {
                **references,
                **event.get_extra("responses_image_generated_references", {}),
            }
        selected_ids = list(dict.fromkeys(reference_ids))
        if any(key not in references for key in selected_ids):
            return "参考图编号不属于本次可用候选；历史生成图仅可用于编辑。目标已不在上下文时，请重新上传或引用图片。"
        if operation == "edit" and (not selected_ids or character_names):
            return "编辑图片须选择本次附件、引用图或最近生成图的候选编号，character_names 请留空；目标不明确时请先确认是哪张图。"
        if self.library is None:
            return "图片插件尚未初始化，请重启 AstrBot 后重试。"
        if event.get_extra("responses_image_delivery_available") is False:
            return (
                "请先启用主 Agent 的 send_message_to_user 工具。"
                "生成图片需要先回传主 Agent 看图，再由该工具发送，本次没有发起生图。"
            )
        original_text = event.get_extra(
            "responses_image_user_text", event.get_message_str()
        )
        if not isinstance(original_text, str) or not original_text.strip():
            return "本次消息没有画图要求，请描述要生成或修改的内容。"
        if len(original_text) > 16000:
            return "本次画图要求过长，请缩短到 16000 字以内。"
        normalized_text = " ".join(original_text.split()).casefold()
        matched_profiles = []
        for name in self.library.list_names():
            candidate = self.library.lookup(name)
            for alias in [candidate["name"], *candidate["aliases"]]:
                normalized_alias = " ".join(alias.split()).casefold()
                appears = (
                    re.search(
                        r"(?<![a-z0-9_])"
                        + re.escape(normalized_alias)
                        + r"(?![a-z0-9_])",
                        normalized_text,
                    )
                    if normalized_alias.isascii()
                    else normalized_alias in normalized_text
                )
                if appears:
                    matched_profiles.append(candidate)
                    break
        targets = {}
        if operation == "new":
            if not character_names:
                if context_ids:
                    return (
                        "已选择上下文，请先解析目标角色并填写 character_names。"
                        "无需让用户重复已经明确的名字。"
                    )
                if len(matched_profiles) > 1:
                    return (
                        "本次提到了多个已建档角色，请在 character_names 中选择实际要画的角色。"
                        "可以选择多人，无需让用户重复名字。"
                    )
                character_names = [profile["name"] for profile in matched_profiles]
            if len(character_names) > 4:
                return "一次最多绘制四个角色，请拆分本次请求。"
            evidence = [normalized_text] + [
                " ".join(text.split()).casefold() for text in selected_context
            ]
            for name in character_names:
                profile = self.library.lookup(name)
                aliases = [profile["name"], *profile["aliases"]] if profile else [name]
                found = False
                for alias in aliases:
                    normalized_alias = " ".join(alias.split()).casefold()
                    for text in evidence:
                        found = bool(
                            re.search(
                                r"(?<![a-z0-9_])"
                                + re.escape(normalized_alias)
                                + r"(?![a-z0-9_])",
                                text,
                            )
                            if normalized_alias.isascii()
                            else normalized_alias in text
                        )
                        if found:
                            break
                    if found:
                        break
                if not found:
                    return (
                        f"当前消息或所选上下文中找不到角色“{name}”的名字依据。"
                        "请选用相关的上下文片段；若指代仍不明确，再确认要画谁。"
                    )
                canonical = profile["name"] if profile else name
                key = " ".join(canonical.split()).casefold()
                targets.setdefault(key, {"name": canonical, "profile": profile})
            # A mention may be a comparison or exclusion, not a requested subject.
            if matched_profiles and not any(
                " ".join(profile["name"].split()).casefold() in targets
                for profile in matched_profiles
            ):
                return "工具选择的角色与本次消息不符，不能用历史角色替换本次明确提及的角色。"
            if selected_ids and any(target["profile"] for target in targets.values()):
                return "本版一次只使用角色库立绘或本次参考图，请勿混用两类参考。"
        missing_targets = [
            target["name"]
            for target in targets.values()
            if not target["profile"] and not selected_ids
        ]
        needs_search = bool(missing_targets)
        search_backend = self.config.get("reference_search", "disabled")
        if search_backend not in {"disabled", "native", "tavily"}:
            return (
                "参考图搜索方式配置无效，请在插件面板选择 disabled、native 或 tavily。"
            )
        if not search_references:
            search_backend = "disabled"
        needs_search = needs_search and search_backend != "disabled"
        generation_api = self.config.get("generation_api", "responses")
        if generation_api not in {"responses", "images"}:
            return "绘图接口配置无效，请在插件面板选择 responses 或 images。"
        provider_id = str(self.config.get("worker_provider_id", "")).strip()
        if not provider_id:
            provider_id = await self.context.get_current_chat_provider_id(
                event.unified_msg_origin
            )
        provider = self.context.get_provider_by_id(provider_id)
        required_type = (
            ProviderOpenAIResponses
            if generation_api == "responses"
            else ProviderOpenAIOfficial
        )
        if not isinstance(provider, required_type):
            return (
                "请在插件配置 worker_provider_id 中选择 openai_responses 类型的模型配置 ID，"
                "或将当前聊天模型切换为 Responses 类型。"
                if generation_api == "responses"
                else "Images API 绘图请在 worker_provider_id 中选择 openai_chat_completion、openai_responses 模型配置。"
            )
        sub_agents = self.config.get("sub_agents_enabled", False) is True
        research_provider = provider
        if sub_agents and needs_search:
            research_id = str(self.config.get("research_provider_id", "")).strip()
            if research_id:
                research_provider = self.context.get_provider_by_id(research_id)
        if needs_search and not isinstance(research_provider, ProviderOpenAIResponses):
            self.logger.warning(
                "Reference research provider unavailable; using text-only generation"
            )
            needs_search = False
            search_backend = "disabled"
        if generation_api == "images":
            provider = ImagesAPIClient(provider)
        else:
            provider = ResponsesImageClient(provider)
        if needs_search:
            research_provider = ResponsesImageClient(research_provider)
        image_model = self.config.get("image_model", "")
        if not isinstance(image_model, str) or len(image_model) > 200:
            return "实际图片模型配置不正确，请填写不超过 200 字符的模型名称。"

        lock = event.get_extra("responses_image_lock")
        if lock is None:
            lock = asyncio.Lock()
            event.set_extra("responses_image_lock", lock)
        async with lock:
            if event.get_extra("responses_image_attempted", False):
                return (
                    "本条消息已经发起过生图流程，请等待结果；如需重试，请发送新消息。"
                )
            event.set_extra("responses_image_requester", event.get_sender_id())
            arguments = {
                "event": event,
                "operation": operation,
                "selected_ids": selected_ids,
                "references": copy.deepcopy(references),
                "targets": copy.deepcopy(targets),
                "original_text": original_text,
                "provider": provider,
                "research_provider": research_provider,
                "sub_agents": sub_agents,
                "search_backend": search_backend,
                "image_model": image_model,
            }
            if self.quota is None:
                return "绘图额度记录尚未初始化，请重启插件。"
            token = "responses-image-" + uuid4().hex
            arguments["quota_token"] = token
            if self.config.get("background_generation", False) is not True:
                try:
                    self.quota.reserve(
                        token, event.unified_msg_origin, admin=event.is_admin()
                    )
                    result = await generation.execute_image_request(self, **arguments)
                    if isinstance(result, CallToolResult):
                        event.set_extra("responses_image_ready", True)
                    return result
                except QuotaExceeded as exc:
                    return str(exc)
                except Exception as exc:
                    self.logger.warning(
                        "Image quota admission failed: %s", type(exc).__name__
                    )
                    return "绘图额度记录不可用，请管理员检查插件配置及日志。"
                finally:
                    self.quota.release(token)
            if self._closing:
                return "图片插件正在停止，暂不接收后台任务。"
            if (
                background.wake_main_agent is None
                and background.run_legacy_completion is None
            ):
                self.logger.debug(
                    "Background image completion unavailable: native_import=%s, legacy_import=%s",
                    background.native_wakeup_import_error,
                    background.legacy_wakeup_import_error,
                )
                return "当前 AstrBot 无法加载后台处理接口，请管理员检查 Debug 日志或关闭后台绘图。"
            if (
                background.wake_main_agent is None
                and self.context.get_config(umo=event.unified_msg_origin)
                .get("agent_runner", {})
                .get("runner_type", "local")
                != "local"
            ):
                return "当前兼容后台绘图需要 AstrBot 本地 Agent，请切换 Agent 类型或关闭后台绘图。"
            cid = event.get_extra("responses_image_conversation_id")
            if not cid:
                return "无法绑定当前会话，请关闭后台绘图后重试。"
            if not getattr(event.platform_meta, "support_proactive_message", False):
                return "当前平台不支持后台主动发送，请关闭后台绘图。"
            admin = event.is_admin()
            submitter = (event.get_platform_id(), event.get_sender_id())
            if not admin and (not submitter[0] or not submitter[1]):
                return "无法识别请求者账号，暂不能提交后台绘图。"
            if not admin and any(
                job["submitter"] == submitter for job in self._jobs.values()
            ):
                return "你已有排队或处理中的绘图请求，请等待该任务完成后再提交。"
            concurrency = self.config.get("max_concurrent_jobs", 4)
            queue_size = self.config.get("max_queue_size", 20)
            if (
                type(concurrency) is not int
                or not 1 <= concurrency <= 16
                or type(queue_size) is not int
                or not 0 <= queue_size <= 200
            ):
                return "队列配置无效：并发数需为 1–16，等待容量需为 0–200。"
            if len(self._jobs) >= concurrency + queue_size:
                return "后台绘图队列已满，请稍后发送新消息重试。"
            try:
                self.quota.reserve(token, event.unified_msg_origin, admin=admin)
            except QuotaExceeded as exc:
                return str(exc)
            except Exception as exc:
                self.logger.warning(
                    "Image quota admission failed: %s", type(exc).__name__
                )
                return "绘图额度记录不可用，请管理员检查插件配置及日志。"
            job = {
                "id": token,
                "umo": event.unified_msg_origin,
                "cid": cid,
                "original_text": original_text,
                "submitter": submitter,
                "state": "queued",
                "created_at": time.time(),
                "started_at": None,
                "operation": operation,
                "quota_exempt": admin,
                "concurrency": concurrency,
                "result": None,
                "claimed": asyncio.Event(),
                "stop": asyncio.Event(),
            }
            self._jobs[token] = job
            event.set_extra("responses_image_attempted", True)
            job["task"] = asyncio.create_task(
                background.run_background_job(self, job, arguments)
            )
            return (
                f"绘图任务已加入后台队列（{token}），可以继续聊天；完成后会自动回到主 Agent 看图并发送。"
                "现在尚未生成图片，请不要声称已完成，也不要重复调用或等待任务。"
            )

    async def terminate(self) -> None:
        """Cancel owned drawing tasks without closing shared model connections."""
        self._closing = True
        jobs = list(self._jobs.values())
        for job in jobs:
            job["stop"].set()
            job["task"].cancel()
        if jobs:
            await asyncio.gather(*(job["task"] for job in jobs), return_exceptions=True)
        for job in jobs:
            if self.quota is not None:
                self.quota.release(job["id"])
        self._jobs.clear()
        routes = getattr(self.context, "registered_web_apis", [])
        routes[:] = [
            route for route in routes if getattr(route[1], "__self__", None) is not self
        ]

    @filter.on_using_llm_tool()
    async def restrict_image_delivery(self, event, tool, tool_args) -> None:
        """Limit image delivery to this chat and its original requester.

        Args:
            event: Event holding the generated image.
            tool: Tool selected by the main agent.
            tool_args: Mutable arguments passed to the tool.
        """
        await delivery.restrict_image_delivery(event, tool, tool_args)

    @filter.on_decorating_result()
    async def image_delivery_reply(self, event: AstrMessageEvent) -> None:
        """Hide stale output after delivery and sanitize final image messages.

        Args:
            event: Event whose final message is being decorated.
        """
        await delivery.image_delivery_reply(event)

    async def queue_status(self) -> dict:
        """Expose bounded queue metadata through AstrBot's authenticated plugin API.

        Returns:
            Task states and positions without prompts, pictures or reference URLs.
        """
        jobs = []
        position = 0
        for job in self._jobs.values():
            if job["state"] == "queued":
                position += 1
            jobs.append(
                {
                    key: job[key]
                    for key in (
                        "id",
                        "umo",
                        "state",
                        "created_at",
                        "started_at",
                        "operation",
                        "quota_exempt",
                    )
                }
                | {"queue_position": position if job["state"] == "queued" else None}
            )
        return {
            "status": "ok",
            "data": {
                "jobs": jobs,
                "queued": position,
                "running": sum(job["state"] == "running" for job in jobs),
                "awaiting_delivery": sum(
                    job["state"] == "awaiting_delivery" for job in jobs
                ),
                "max_concurrent_jobs": self.config.get("max_concurrent_jobs", 4),
                "max_queue_size": self.config.get("max_queue_size", 20),
            },
        }

    @filter.llm_tool(name="responses_image_queue")
    async def query_image_queue(self, event: AstrMessageEvent, window: str = "") -> str:
        """Read the drawing queue to answer natural-language status questions.

        This is read-only and never starts generation. Default to the current chat.
        Use is_mine to identify the caller's jobs. can_cancel describes current
        eligibility only; cancellation rechecks it. Administrators also receive
        requester account IDs to identify others' jobs. Positions are global waiting
        order, not an ETA. Synchronous drawing is not listed in this queue.

        Args:
            window(string): Empty for the current chat. Only administrators may supply another exact window UMO or all to inspect every window. Never invent a UMO.

        Returns:
            JSON with visible job states, counts and caller ownership, or a safe error.
        """
        if not isinstance(window, str) or len(window) > 512:
            return "窗口参数不正确；查询当前窗口请留空。"
        target = window.strip() or event.unified_msg_origin
        admin = event.is_admin()
        if target != event.unified_msg_origin and not admin:
            return "只有管理员可以查询其他窗口的绘图队列。"
        if target != "all" and target.count(":") < 2:
            return "请使用完整窗口 UMO；查询当前窗口请留空。"
        snapshot = (await self.queue_status())["data"]
        caller = (event.get_platform_id(), event.get_sender_id())
        jobs = []
        for job in snapshot["jobs"]:
            if target != "all" and job["umo"] != target:
                continue
            submitter = self._jobs.get(job["id"], {}).get("submitter")
            is_mine = bool(all(caller) and submitter == caller)
            visible = job | {
                "is_mine": is_mine,
                "can_cancel": job["state"] == "queued" and (admin or is_mine),
            }
            if admin and submitter:
                visible["requester"] = {
                    "platform_id": submitter[0],
                    "user_id": submitter[1],
                }
            jobs.append(visible)
        return json.dumps(
            {
                "status": "ok",
                "data": {
                    "window": target,
                    "background_enabled": self.config.get(
                        "background_generation", False
                    )
                    is True,
                    "jobs": jobs,
                    "total": len(jobs),
                    "queued": sum(job["state"] == "queued" for job in jobs),
                    "running": sum(job["state"] == "running" for job in jobs),
                    "awaiting_delivery": sum(
                        job["state"] == "awaiting_delivery" for job in jobs
                    ),
                    "queue_position_scope": "global_waiting_order",
                },
            },
            ensure_ascii=False,
        )

    @filter.llm_tool(name="responses_image_quota")
    async def query_image_quota(self, event: AstrMessageEvent, window: str = "") -> str:
        """Read remaining drawing allowance to answer natural-language questions.

        Counts are requests shared by a chat window over the last hour and 24 hours,
        not calendar-day counts or a personal allowance. Remaining values subtract
        both used requests and reservations; null means unlimited. The caller's
        allowance is unlimited for administrators even when the window is exhausted.
        This does not reserve, spend, reset or guarantee admission to the queue.

        Args:
            window(string): Empty for the current chat; administrators may query another exact window UMO. Aggregate all-window quota queries are not supported. Never invent a UMO.

        Returns:
            JSON with window usage, remaining requests and caller exemption, or a safe error.
        """
        if not isinstance(window, str) or len(window) > 512:
            return "窗口参数不正确；查询当前窗口请留空。"
        target = window.strip() or event.unified_msg_origin
        admin = event.is_admin()
        if target != event.unified_msg_origin and not admin:
            return "只有管理员可以查询其他窗口的绘图额度。"
        if target.count(":") < 2:
            return "额度按窗口统计，请使用完整窗口 UMO；查询当前窗口请留空。"
        if self.quota is None:
            return "绘图额度记录尚未初始化。"
        try:
            status = self.quota.snapshot(target)
        except Exception as exc:
            self.logger.warning("Drawing quota query failed: %s", type(exc).__name__)
            return "绘图额度暂不可查询，请管理员检查插件配置及日志。"
        remaining_hour = (
            max(0, status["hourly_limit"] - status["used_hour"] - status["reserved"])
            if status["hourly_limit"]
            else None
        )
        remaining_day = (
            max(0, status["daily_limit"] - status["used_day"] - status["reserved"])
            if status["daily_limit"]
            else None
        )
        remaining = min(
            (value for value in (remaining_hour, remaining_day) if value is not None),
            default=None,
        )
        caller = (event.get_platform_id(), event.get_sender_id())
        return json.dumps(
            {
                "status": "ok",
                "data": status
                | {
                    "unit": "drawing_requests",
                    "window_type": "rolling",
                    "remaining_hour": remaining_hour,
                    "remaining_day": remaining_day,
                    "remaining_requests": remaining,
                    "caller_quota_exempt": admin,
                    "caller_remaining_requests": None if admin else remaining,
                    "caller_has_pending_background_task": bool(all(caller))
                    and any(job["submitter"] == caller for job in self._jobs.values()),
                },
            },
            ensure_ascii=False,
        )

    @filter.llm_tool(name="responses_image_cancel")
    async def cancel_queued_image(
        self, event: AstrMessageEvent, job_id: str = ""
    ) -> str:
        """Cancel queued drawing work only on the current user's explicit request.

        Members may cancel only their own tasks, matched by platform and sender ID.
        Administrators may cancel anyone's queued task using its exact ID. Empty
        job_id selects only the caller's unique queued task, across chat windows on
        the same platform account. Ambiguous selections never cancel anything.
        Tasks already running, including reference research, cannot be cancelled.
        Success releases reservations but never refunds previously spent quota.

        Args:
            job_id(string): Exact task ID from a submission receipt or queue query. Empty selects the caller's unique queued task. Never invent IDs or use a queue position as an ID.

        Returns:
            JSON confirming cancellation and released reservations, or a safe refusal.
        """
        if not isinstance(job_id, str) or len(job_id) > 128:
            return "任务编号不正确，请使用队列中的完整任务 ID。"
        job_id = job_id.strip()
        admin = event.is_admin()
        caller = (event.get_platform_id(), event.get_sender_id())
        if not all(caller) and (not admin or not job_id):
            return "无法识别请求者账号；管理员可指定完整任务 ID 取消排队任务。"
        async with self._queue_changed:
            if self._closing:
                return "图片插件正在停止，暂不能操作队列。"
            if job_id:
                job = self._jobs.get(job_id)
                if job is None or (not admin and job["submitter"] != caller):
                    return "任务不存在、已结束，或你无权取消该任务。"
            else:
                candidates = [
                    item
                    for item in self._jobs.values()
                    if item["submitter"] == caller and item["state"] == "queued"
                ]
                if not candidates:
                    return "没有找到你仍在排队的绘图任务；已开始执行的任务不能取消。"
                if len(candidates) != 1:
                    return "你有多个排队任务，请先查询队列并明确要取消的任务 ID；本次没有取消任何任务。"
                job = candidates[0]
                job_id = job["id"]
            if job["state"] != "queued":
                return (
                    "该任务已开始执行或正在等待回传，不能作为排队任务取消；额度未变。"
                )
            released = int(self.quota is not None and job_id in self.quota.reservations)
            # A task cancelled before its first step will never execute its finally.
            job["stop"].set()
            job["state"] = "cancelled"
            job["result"] = None
            self._jobs.pop(job_id)
            if self.quota is not None:
                self.quota.release(job_id)
            task = job["task"]
            task.cancel()
            self._queue_changed.notify_all()
        # The worker's finally acquires the condition; never join it under the lock.
        await asyncio.gather(task, return_exceptions=True)
        self.logger.debug(
            "Queued image task cancelled; released_reservations=%s", released
        )
        return json.dumps(
            {
                "status": "ok",
                "data": {
                    "job_id": job_id,
                    "state": "cancelled",
                    "released_reservations": released,
                },
            },
            ensure_ascii=False,
        )

    @filter.command("生图队列")
    async def inspect_queue(self, event: AstrMessageEvent, window: GreedyStr = ""):
        """List current-window tasks; administrators may inspect another window.

        Args:
            event: Caller whose permissions bound the query.
            window: Optional unified message origin, or all for administrators.

        Yields:
            Queue state without private task contents.
        """
        target = str(window).strip() or event.unified_msg_origin
        if target != event.unified_msg_origin and not event.is_admin():
            yield event.plain_result("只有管理员可以查询其他窗口的绘图队列。")
            return
        snapshot = (await self.queue_status())["data"]
        states = {
            "queued": "排队",
            "running": "执行中",
            "awaiting_delivery": "等待主 Agent 领取",
        }
        rows = [
            f"{job['id']} | {states[job['state']]}"
            + (f" | 排队序号 {job['queue_position']}" if job["queue_position"] else "")
            + (f" | {job['umo']}" if target == "all" else "")
            for job in snapshot["jobs"]
            if target == "all" or job["umo"] == target
        ]
        yield event.plain_result("\n".join(rows) if rows else "该窗口暂无绘图任务。")

    @filter.command("生图额度")
    async def inspect_quota(self, event: AstrMessageEvent, window: GreedyStr = ""):
        """Show rolling usage including queue reservations.

        Args:
            event: Caller whose permissions bound the query.
            window: Optional administrator-selected unified message origin.

        Yields:
            Current usage, limits and the queried window identifier.
        """
        target = str(window).strip() or event.unified_msg_origin
        if target != event.unified_msg_origin and not event.is_admin():
            yield event.plain_result("只有管理员可以查询其他窗口的绘图额度。")
            return
        if self.quota is None:
            yield event.plain_result("绘图额度记录尚未初始化。")
            return
        status = self.quota.snapshot(target)
        yield event.plain_result(
            f"窗口：{target}\n最近一小时：{status['used_hour']}/{status['hourly_limit'] or '不限'} 次；"
            f"最近 24 小时：{status['used_day']}/{status['daily_limit'] or '不限'} 次；"
            f"排队预留：{status['reserved']} 次（同时占用小时和每天额度）。管理员免额。"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("生图限额")
    async def manage_quota(self, event: AstrMessageEvent, arguments: GreedyStr = ""):
        """Override window limits or reset committed usage without releasing jobs.

        Args:
            event: Administrator request; role is checked again at execution.
            arguments: Two limits and optional UMO, or reset and optional UMO.

        Yields:
            Confirmation or command usage.
        """
        if not event.is_admin():
            yield event.plain_result("只有管理员可以设置或重置绘图限额。")
            return
        parts = str(arguments).split()
        try:
            if parts and parts[0].lower() == "reset" and len(parts) in {1, 2}:
                target = parts[1] if len(parts) == 2 else event.unified_msg_origin
                limits = None
            elif len(parts) in {2, 3}:
                limits = [int(parts[0]), int(parts[1])]
                if min(limits) < 0:
                    raise ValueError("Negative drawing limit")
                target = parts[2] if len(parts) == 3 else event.unified_msg_origin
            else:
                raise ValueError("Invalid drawing quota command")
            if len(target) > 512 or target.count(":") < 2:
                raise ValueError("Invalid unified message origin")
            if self.quota is None:
                raise RuntimeError("Drawing quota is unavailable")
            status = self.quota.update(target, limits=limits)
        except ValueError:
            yield event.plain_result(
                "用法：/生图限额 小时次数 每天次数 [窗口 UMO]；/生图限额 reset [窗口 UMO]。0 表示不限；省略窗口表示当前窗口。"
            )
            return
        except Exception as exc:
            self.logger.warning("Drawing quota update failed: %s", type(exc).__name__)
            yield event.plain_result("额度修改未保存，请检查插件数据目录及日志。")
            return
        yield event.plain_result(
            ("已重置已用额度" if limits is None else "已更新窗口限额")
            + f"：{target}。小时 {status['hourly_limit'] or '不限'}，每天 {status['daily_limit'] or '不限'}；"
            + f"保留 {status['reserved']} 次排队预留。"
        )

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("生图建档")
    async def save_character(self, event: AstrMessageEvent, names: GreedyStr):
        """Register an administrator-selected character image and its aliases.

        Args:
            event: Administrator message containing exactly one reference image.
            names: Canonical character name and optional aliases separated by pipes.

        Yields:
            Registration result or an actionable input error.
        """
        if not event.is_admin():
            yield event.plain_result("只有管理员可以维护角色参考库。")
            return
        references = collect_event_images(event)
        if len(references) != 1:
            yield event.plain_result(
                "请附带或引用恰好一张确认正确的立绘，再发送 /生图建档 角色名|别名。"
            )
            return
        parts = [part.strip() for part in str(names).split("|") if part.strip()]
        if not parts or self.library is None:
            yield event.plain_result("请填写角色名，并确认图片插件已初始化。")
            return
        try:
            profile = await self.library.save(
                parts[0], parts[1:], next(iter(references.values()))["image"]
            )
        except Exception as exc:
            self.logger.warning("Character registration failed: %s", type(exc).__name__)
            yield event.plain_result(
                "角色建档失败，请检查图片格式、大小，以及名字或别名是否与其他角色重复。"
            )
            return
        yield event.plain_result(
            f"已保存角色“{profile['name']}”的参考图。后续可以直接使用角色名或别名请求画图。"
        )

    @filter.command("生图角色列表")
    async def list_characters(self, event: AstrMessageEvent):
        """List registered character names without exposing their file paths.

        Args:
            event: The message requesting the character catalog.

        Yields:
            The registered names or setup instructions.
        """
        names = self.library.list_names() if self.library else []
        yield event.plain_result(
            "已建档角色：\n" + "\n".join(names)
            if names
            else "角色库为空。管理员可附图发送 /生图建档 角色名|别名。"
        )
