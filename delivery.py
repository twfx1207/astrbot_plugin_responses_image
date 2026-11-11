"""Finish image delivery through AstrBot's normal direct-result tool path."""

import random
import re
from contextlib import aclosing
from html import escape

from astrbot.api import message_components as Comp
from astrbot.api.event import MessageChain
from astrbot.core.agent.tool import FunctionTool
from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
from astrbot.core.platform.message_session import MessageSession
from astrbot.core.platform.message_type import MessageType
from mcp.types import CallToolResult, TextContent

IMAGE_DELIVERY_INSTRUCTIONS = (
    "Send all generated images together to this session in one call to "
    "send_message_to_user. "
    "You may include one brief plain-text explanation, up to 200 characters, "
    "with the images. Do not include reference sources, links or internal paths. "
    "The plugin adds the original requester's mention in group chats; do not "
    "choose or add other mentions. A successful image send ends this turn "
    "automatically, without another model call or a final confirmation. Put any "
    "brief explanation in the sending tool's messages before the images. "
    "If delivery fails, explain the error briefly without generating again. "
)
CAPTION_LINK_OR_MENTION = re.compile(
    r"https?://|www\.|data:|file://|<qqbot-at-|<@|\[CQ:at|@(?:everyone|here)\b",
    re.IGNORECASE,
)


MARKDOWN_UNAVAILABLE = re.compile(
    r"\b(?:50056|40034090)\b|不允许发送(?:原生\s*)?\s*markdown|"
    r"not allowed to send markdown|markdown[^\n]*(?:not allowed|permission denied)",
    re.IGNORECASE,
)


class ImageDeliveryTool(FunctionTool):
    """Wrap one request's sending tool without modifying the shared tool registry."""

    def __init__(self, original: FunctionTool, plugin) -> None:
        """Keep the existing schema and delegate validation and sending to core.

        Args:
            original: Sending tool from this request's enabled tools.
            plugin: Plugin owning the delivery policy and logger.
        """
        super().__init__(
            name=original.name,
            description=original.description,
            parameters=original.parameters,
            active=original.active,
        )
        self.original = original
        self.plugin = plugin

    async def call(self, context, **kwargs):
        """End the loop only after the image sender reports successful delivery.

        Args:
            context: Current AstrBot agent execution context.
            **kwargs: Sending arguments, filtered by the plugin's delivery hook.

        Yields:
            Original results on failure or ordinary sends; None after image success
            signals normal completion to AstrBot without another model request.
        """
        event = context.context.event
        ready = event.get_extra("responses_image_ready")
        if ready and event.get_extra("responses_image_delivered"):
            event.clear_result()
            yield None
            return
        if ready:
            # Revalidate here as well so the per-request wrapper is self-contained.
            await restrict_image_delivery(event, self, kwargs)
        messages = kwargs.get("messages") or []
        has_images = ready and any(
            isinstance(part, dict) and part.get("type") == "image" for part in messages
        )
        notice = []
        if (
            has_images
            and event.get_message_type() == MessageType.GROUP_MESSAGE
            and event.get_platform_name() in {"qq_official", "qq_official_webhook"}
            and any(part.get("type") == "mention_user" for part in messages)
        ):
            # QQ media content cannot render mentions, but can carry plain text.
            notice = [part for part in messages if part.get("type") != "image"]
            kwargs["messages"] = [
                part for part in messages if part.get("type") == "image"
            ]
            session = MessageSession.from_str(event.unified_msg_origin)
            platform = self.plugin.context.get_platform_inst(session.platform_id)
            target_id = session.session_id.rsplit("_", 1)[-1]
            if (
                getattr(platform, "_session_scene", {}).get(target_id) == "group"
                and getattr(platform, "use_markdown_default", True) is False
            ):
                caption = next(
                    (part["text"] for part in notice if part["type"] == "plain"),
                    "图片已生成。",
                )
                kwargs["messages"].insert(0, {"type": "plain", "text": caption})
                notice = []
        executor = FunctionToolExecutor.execute(
            tool=self.original, run_context=context, **kwargs
        )
        delivered = False
        async with aclosing(executor):
            async for result in executor:
                if has_images and result is None:
                    # An absent receipt is not evidence that an image was sent.
                    yield CallToolResult(
                        isError=True,
                        content=[
                            TextContent(
                                type="text", text="Image delivery was not confirmed."
                            )
                        ],
                    )
                    continue
                confirmed = (
                    has_images
                    and isinstance(result, CallToolResult)
                    and not result.isError
                    and any(
                        isinstance(part, TextContent)
                        and part.text
                        == f"Message sent to session {event.unified_msg_origin}"
                        for part in result.content
                    )
                )
                if not confirmed:
                    yield result
                    continue
                delivered = True
                event.set_extra("responses_image_delivered", True)
        if not delivered:
            return
        # Yield only the terminal signal: yielding a receipt as well would create
        # two tool-result messages for the same call ID in core's history. Signal
        # success before the notification so its timeout cannot reopen the loop.
        event.clear_result()
        yield None
        if notice:
            try:
                await self._send_qq_notification(event, notice)
            except Exception as exc:
                # A failed notification must not resend an already delivered image.
                self.plugin.logger.warning(
                    "Image delivered but requester notification failed: %s",
                    type(exc).__name__,
                )

    async def _send_qq_notification(self, event, notice: list[dict]) -> None:
        """Use supported QQ mentions without leaking tags through core's fallback.

        Args:
            event: Original or background completion event for this chat.
            notice: Validated requester mention and optional short caption.

        Raises:
            RuntimeError: The platform or a successful send receipt is unavailable.
            Exception: An unrelated API failure, including an uncertain timeout.
        """
        caption = next(
            (part["text"] for part in notice if part["type"] == "plain"),
            "图片已生成。",
        )
        requester = next(
            part["mention_user_id"] for part in notice if part["type"] == "mention_user"
        )
        session = MessageSession.from_str(event.unified_msg_origin)
        platform = self.plugin.context.get_platform_inst(session.platform_id)
        # Match the official connector's group-session normalization, including
        # member-isolated sessions. Never derive a destination from model input.
        target_id = session.session_id.rsplit("_", 1)[-1]
        scene = getattr(platform, "_session_scene", {}).get(target_id)
        chain = [Comp.Plain(caption)]
        if scene in {"channel", "guild"}:
            # Official text subchannels support mentions in ordinary content.
            chain.insert(0, Comp.At(qq=requester))
        elif scene == "group" and getattr(platform, "use_markdown_default", True):
            send = getattr(
                getattr(getattr(platform, "client", None), "api", None),
                "post_group_message",
                None,
            )
            message_id = getattr(platform, "_session_last_message_id", {}).get(
                target_id
            )
            proactive = getattr(platform, "_allow_group_proactive_send", False)
            if callable(send) and (message_id or proactive):
                payload = {
                    "group_openid": target_id,
                    "msg_type": 2,
                    "markdown": {
                        "content": f'<qqbot-at-user id="{escape(requester, quote=True)}" /> {caption}'
                    },
                    "msg_seq": random.randint(1, 10000),
                }
                if message_id and not proactive:
                    payload["msg_id"] = message_id
                try:
                    # Use the existing SDK connection, bypassing only core's
                    # Markdown-to-content fallback which retains raw mention tags.
                    response = await send(**payload)
                except Exception as exc:
                    if not MARKDOWN_UNAVAILABLE.search(str(exc)):
                        raise
                    self.plugin.logger.debug(
                        "QQ image notification using plain text: markdown_unavailable"
                    )
                else:
                    receipt_id = (
                        response.get("id")
                        if isinstance(response, dict)
                        else getattr(response, "id", None)
                    )
                    if not receipt_id:
                        raise RuntimeError("QQ notification delivery was not confirmed")
                    platform.remember_session_message_id(target_id, str(receipt_id))
                    return
        # Group mentions cannot be represented in ordinary content. Unknown
        # capabilities also take this path; no speculative tags are ever emitted.
        sent = await self.plugin.context.send_message(
            event.unified_msg_origin,
            MessageChain(chain=chain, use_markdown_=False),
        )
        if not sent:
            raise RuntimeError("Notification platform is unavailable")


async def restrict_image_delivery(event, tool, tool_args) -> None:
    """Attach the trusted group requester and allow one short explanation.

    Args:
        event: Main-agent event holding the completed image result.
        tool: Tool about to execute.
        tool_args: Mutable arguments consumed by AstrBot's executor.
    """
    if (
        event.get_extra("responses_image_ready")
        and tool.name == "send_message_to_user"
        and isinstance(tool_args, dict)
    ):
        messages = tool_args.get("messages")
        if isinstance(messages, list):
            images = [
                message
                for message in messages
                if isinstance(message, dict) and message.get("type") == "image"
            ]
            delivery = []
            if images:
                requester = event.get_extra("responses_image_requester")
                if (
                    event.get_message_type() == MessageType.GROUP_MESSAGE
                    and isinstance(requester, (str, int))
                    and str(requester).strip()
                    and str(requester).lower() != "all"
                ):
                    delivery.append(
                        {"type": "mention_user", "mention_user_id": str(requester)}
                    )
                for message in messages:
                    if not isinstance(message, dict) or message.get("type") != "plain":
                        continue
                    caption = message.get("text")
                    if (
                        isinstance(caption, str)
                        and caption.strip()
                        and not CAPTION_LINK_OR_MENTION.search(caption)
                    ):
                        delivery.append(
                            {"type": "plain", "text": caption.strip()[:200]}
                        )
                        break
                delivery.extend(images)
            # Delivery belongs to the requesting chat, including for admins.
            tool_args.pop("session", None)
            tool_args["messages"] = delivery


async def image_delivery_reply(event) -> None:
    """Suppress stale output after a confirmed image send.

    Args:
        event: Event whose final non-streaming result is being decorated.
    """
    if not event.get_extra("responses_image_ready"):
        return
    result = event.get_result()
    if result and (
        event.get_extra("responses_image_delivered")
        or any(isinstance(part, Comp.Image) for part in result.chain)
    ):
        event.set_extra("enable_reasoning", False)
        event.set_extra("_llm_reasoning_content", "")
        if event.get_extra("responses_image_delivered"):
            result.chain = []
            return
        images = [part for part in result.chain if isinstance(part, Comp.Image)]
        delivery = []
        qq_official = event.get_platform_name() in {
            "qq_official",
            "qq_official_webhook",
        }
        requester = event.get_extra("responses_image_requester")
        if (
            event.get_message_type() == MessageType.GROUP_MESSAGE
            and not qq_official
            and isinstance(requester, (str, int))
            and str(requester).strip()
            and str(requester).lower() != "all"
        ):
            delivery.append(Comp.At(qq=str(requester)))
        for part in result.chain:
            if (
                isinstance(part, Comp.Plain)
                and part.text.strip()
                and not CAPTION_LINK_OR_MENTION.search(part.text)
            ):
                delivery.append(Comp.Plain(part.text.strip()[:200]))
                break
        # A final image chain bypasses the controlled sending tool. Keep its
        # QQ fallback tag-free because media content cannot render mentions.
        if qq_official and not delivery:
            delivery.append(Comp.Plain("图片已生成。"))
        result.chain = delivery + images
