"""Run completion turns on AstrBot releases predating wake_main_agent."""

import asyncio
import copy
import json
from contextlib import aclosing

from astrbot.core.astr_main_agent import MainAgentBuildConfig, build_main_agent
from astrbot.core.config.agent_runner import resolve_context_compression_config
from astrbot.core.cron.events import CronMessageEvent
from astrbot.core.log import log_context
from astrbot.core.pipeline.context_utils import call_event_hook
from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
    InternalAgentSubStage,
)
from astrbot.core.platform.astr_message_event import event_context
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.provider.entities import ProviderRequest
from astrbot.core.star.session_llm_manager import SessionServiceManager
from astrbot.core.star.star_handler import EventType
from astrbot.core.utils.active_event_registry import active_event_registry
from astrbot.core.utils.config_number import coerce_int_config
from astrbot.core.utils.session_lock import session_lock_manager
from astrbot.core.utils.trace import trace_execution


class LegacyCompletionEvent(CronMessageEvent):
    """Use the existing proactive sender without reusing an incoming QQ event."""

    def __init__(self, context, job: dict, prompt: str) -> None:
        """Bind a fresh, ordinary-permission event to the original destination.

        Args:
            context: AstrBot services, including the proactive sender.
            job: Original session, task ID and cancellation signal.
            prompt: Fixed completion instructions, without generated content.

        Raises:
            RuntimeError: If the original platform is no longer available.
        """
        session = MessageSession.from_str(job["umo"])
        platform = context.get_platform_inst(session.platform_id)
        if platform is None or not platform.meta().support_proactive_message:
            raise RuntimeError("Background completion platform unavailable")
        super().__init__(
            context=context,
            session=session,
            message=prompt,
            message_type=session.message_type,
        )
        # Preserve the real platform for persona selection and scoped settings.
        self.platform_meta = platform.meta()
        self.source = "plugin"
        self.source_id = job["id"]
        self.role = "member"
        self.set_extra("_background_stop_signal", job["stop"])
        self.set_extra("enable_streaming", False)


async def run_legacy_completion(plugin, job: dict, prompt: str) -> None:
    """Build and run a main-agent turn inside the existing background task.

    This uses the released cron-style entry path, with the same session lock,
    request hooks and multimodal history saver as normal chat. It is called only
    when the newer wakeup imports are unavailable, never after a native failure.

    Args:
        plugin: Owner of the queued job and its validated result retrieval hook.
        job: Session-bound result awaiting review and delivery.
        prompt: Fixed completion instructions.

    Raises:
        RuntimeError: If the main agent cannot be constructed or fails to finish.
        asyncio.CancelledError: If the plugin stops or the completion is cancelled.
    """
    context = plugin.context
    event = LegacyCompletionEvent(context, job, prompt)
    runner = None
    runner_started = False
    reset = None
    with (
        log_context(event.get_platform_id(), event.get_platform_name()),
        event_context(event),
        trace_execution(event.trace),
    ):
        async with session_lock_manager.acquire_lock(job["umo"]):
            # Read history only after the submission turn has saved its receipt.
            if plugin._closing or job["stop"].is_set() or event.is_stopped():
                return
            cfg = context.get_config(umo=job["umo"])
            selected = cfg.get("plugin_set", ["*"])
            event.plugins_name = None if selected == ["*"] else selected
            if (
                (selected != ["*"] and "astrbot_plugin_responses_image" not in selected)
                or not cfg.get("provider_settings", {}).get("enable", True)
                or cfg.get("agent_runner", {}).get("runner_type", "local") != "local"
                or not await SessionServiceManager.should_process_llm_request(event)
            ):
                plugin.logger.debug("Legacy image completion skipped: session_disabled")
                return
            cid = await context.conversation_manager.get_curr_conversation_id(
                job["umo"]
            )
            if cid != job["cid"]:
                plugin.logger.debug(
                    "Legacy image completion skipped: conversation_changed"
                )
                return
            conversation = await context.conversation_manager.get_conversation(
                job["umo"], cid
            )
            if conversation is None:
                return
            request = ProviderRequest(
                prompt=prompt,
                conversation=conversation,
                contexts=json.loads(conversation.history or "[]"),
            )
            settings = copy.deepcopy(cfg.get("provider_settings", {}))
            agent_config = cfg.get("agent_runner", {}).get("config", {})
            persona = agent_config.get("persona", {})
            model = agent_config.get("model", {})
            misc = agent_config.get("misc", {})
            settings["default_personality"] = persona.get("persona_id", "default")
            config = MainAgentBuildConfig(
                tool_call_timeout=misc.get("tool_call_timeout", 120),
                streaming_response=False,
                tool_schema_mode="full",
                **resolve_context_compression_config(
                    agent_config.get("compression", {})
                ),
                llm_safety_mode=persona.get("safety_mode", True),
                safety_mode_strategy=persona.get(
                    "safety_mode_strategy", "system_prompt"
                ),
                provider_settings=settings,
                fallback_provider_ids=model.get("fallback_provider_ids", []),
                request_max_retries=model.get("request_max_retries", 5),
                add_cron_tools=False,
                timezone=cfg.get("timezone"),
            )
            result = await build_main_agent(
                event=event,
                plugin_context=context,
                config=config,
                req=request,
                apply_reset=False,
            )
            if result is None:
                raise RuntimeError("Legacy image completion agent build failed")
            runner = result.agent_runner
            reset = result.reset_coro
            request = result.provider_request
            try:
                if await call_event_hook(event, EventType.OnLLMRequestEvent, request):
                    return
                # Enforce the receipt and tool allowlist even if a hook is absent
                # or another plugin changed the request later in hook dispatch.
                await plugin.prepare_image_request(event, request)
                if event.is_stopped() or plugin._closing or job["stop"].is_set():
                    return
                if reset is not None:
                    pending_reset, reset = reset, None
                    await pending_reset
                runner_started = True
                active_event_registry.register_agent_stop_callback(
                    event, runner.request_stop
                )
                steps = coerce_int_config(
                    misc.get("max_steps", 128),
                    default=128,
                    min_value=1,
                    field_name="agent_runner.config.misc.max_steps",
                )
                async with aclosing(runner.step_until_done(steps)) as results:
                    async for _ in results:
                        if (
                            plugin._closing
                            or job["stop"].is_set()
                            or event.is_stopped()
                        ):
                            runner.request_stop()
                            return
                if runner.was_aborted() or job["stop"].is_set() or event.is_stopped():
                    return
                if runner.state.name != "DONE":
                    raise RuntimeError("Legacy image completion agent failed")
                # The cron summary saver drops tool images. Reuse chat's saver
                # so pixels and tool calls survive into a later editing request.
                history_stage = InternalAgentSubStage()
                history_stage.conv_manager = context.conversation_manager
                await history_stage._save_to_history(
                    event,
                    request,
                    runner.get_final_llm_resp(),
                    runner.run_context.messages,
                    runner.stats,
                )
                if not job["claimed"].is_set():
                    plugin.logger.warning(
                        "Legacy image completion did not retrieve its result"
                    )
                elif (
                    (
                        event.get_extra("responses_image_delivered")
                        or not event._has_send_oper
                    )
                    and (response := runner.get_final_llm_resp()) is not None
                    and response.completion_text
                ):
                    event.set_result(event.plain_result(response.completion_text))
                    await plugin.image_delivery_reply(event)
                    reply = event.get_result()
                    if reply and reply.chain:
                        await event.send(reply)
            except asyncio.CancelledError:
                job["stop"].set()
                raise
            finally:
                active_event_registry.unregister_agent_stop_callback(event)
                if reset is not None:
                    reset.close()
                if runner_started and not runner.done():
                    runner.request_stop()
