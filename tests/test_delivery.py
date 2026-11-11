"""Check terminal image sends, failure recovery and connector-aware mentions."""

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from astrbot.core.message import components as Comp
from astrbot.core.message.message_event_result import MessageEventResult
from astrbot.core.provider.entities import ProviderRequest
from mcp.types import CallToolResult, TextContent
from test_control import FakeEvent
from test_control import control_env as control_env


@pytest.mark.asyncio
@pytest.mark.parametrize("requester", ["original-user", "all", "", None])
async def test_group_delivery_uses_trusted_requester_without_duplicate_mentions(
    control_env, requester
):
    env = control_env
    event = FakeEvent(admin=True)
    event.unified_msg_origin = "qq:GroupMessage:room"
    event.set_extra("responses_image_ready", True)
    event.set_extra("responses_image_requester", requester)
    event.sender_id = "background-agent"
    args = {
        "session": "qq:GroupMessage:unrelated-group",
        "messages": [
            {"type": "mention_user", "mention_user_id": "wrong-person"},
            {"type": "mention_user", "mention_user_id": "all"},
            {"type": "plain", "text": "source https://reference.example"},
            {"type": "plain", "text": '<qqbot-at-user id="wrong-person" />'},
            {"type": "plain", "text": "说明" * 150},
            {"type": "plain", "text": "duplicate explanation"},
            {"type": "image", "path": "/generated.png"},
        ],
    }
    tool = SimpleNamespace(name="send_message_to_user")
    await env.plugin.restrict_image_delivery(event, tool, args)
    first = copy.deepcopy(args)
    await env.plugin.restrict_image_delivery(event, tool, args)
    assert args == first and "session" not in args
    expected = (
        [{"type": "mention_user", "mention_user_id": requester}]
        if requester == "original-user"
        else []
    )
    assert args["messages"] == expected + [
        {"type": "plain", "text": "说明" * 100},
        {"type": "image", "path": "/generated.png"},
    ]
    env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("group", [False, True])
@pytest.mark.parametrize("background", [False, True])
async def test_onebot_v11_uses_native_at_and_image_segments(
    control_env, monkeypatch, tmp_path, group, background
):
    import ast
    import base64
    import logging
    from pathlib import Path

    import astrbot
    from astrbot.core.agent.run_context import ContextWrapper
    from astrbot.core.message.message_event_result import MessageChain
    from astrbot.core.platform.message_session import MessageSession
    from astrbot.core.platform.message_type import MessageType
    from astrbot.core.tools.message_tools import SendMessageToUserTool

    env = control_env
    namespace = dict(vars(Comp))
    namespace.update(
        MessageChain=MessageChain,
        MessageType=MessageType,
        Event=type("OfflineEvent", (dict,), {}),
        Platform=type("OfflinePlatform", (), {"send_by_session": AsyncMock()}),
        logger=logging.getLogger("onebot-wire-test"),
    )
    core = Path(astrbot.__file__).parent / "core/platform/sources/aiocqhttp"
    for file_name, class_name, methods in (
        (
            "aiocqhttp_message_event.py",
            "AiocqhttpMessageEvent",
            {
                "_from_segment_to_dict",
                "_parse_onebot_json",
                "_dispatch_send",
                "send_message",
            },
        ),
        ("aiocqhttp_platform_adapter.py", "AiocqhttpAdapter", {"send_by_session"}),
    ):
        source = ast.parse((core / file_name).read_text(encoding="utf-8"))
        cls = next(
            node
            for node in source.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        cls.body = [
            node
            for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in methods
        ]
        if class_name == "AiocqhttpMessageEvent":
            cls.bases = []
        cls.decorator_list = []
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                cls,
            ],
            type_ignores=[],
        )
        exec(
            compile(ast.fix_missing_locations(module), str(core / file_name), "exec"),
            namespace,
        )
    adapter = namespace["AiocqhttpAdapter"]()
    adapter.bot = SimpleNamespace(
        send_group_msg=AsyncMock(), send_private_msg=AsyncMock()
    )

    async def send_message(session, chain):
        """Run the real OneBot routing and message serialization methods.

        Args:
            session: Conversation identifier selected by core's sending tool.
            chain: Filtered requester mention, caption and generated image.

        Returns:
            True after the mocked protocol endpoint returns.
        """
        await adapter.send_by_session(MessageSession.from_str(str(session)), chain)
        return True

    env.context.send_message = AsyncMock(side_effect=send_message)
    env.context.get_platform_inst = Mock(
        side_effect=AssertionError("OneBot must not query QQ Official capabilities")
    )
    event = FakeEvent()
    event.platform_name = "aiocqhttp"
    event.platform_id = "napcat-main"
    event.unified_msg_origin = (
        "napcat-main:GroupMessage:10001_778899"
        if group
        else "napcat-main:FriendMessage:10001"
    )
    event.sender_id = "20002" if background else "10001"
    event.set_extra("responses_image_requester", "10001")
    event.set_extra("responses_image_ready", True)
    event.clear_result = Mock()
    event.get_result = Mock(return_value=None)
    original = SendMessageToUserTool()
    image_path = tmp_path / "generated.png"
    # File authorization is covered by core integration tests; keep this test on
    # the actual send tool, connector routing and JSON serialization boundary.
    monkeypatch.setattr(
        original,
        "_resolve_path_from_sandbox",
        AsyncMock(return_value=(str(image_path), False)),
    )
    wrapper = env.module.delivery.ImageDeliveryTool(original, env.plugin)
    arguments = {
        "session": "unrelated:GroupMessage:99999",
        "messages": [
            {"type": "mention_user", "mention_user_id": "all"},
            {"type": "mention_user", "mention_user_id": "20002"},
            {"type": "plain", "text": '<qqbot-at-user id="20002" />'},
            {"type": "plain", "text": "[CQ:at,qq=20002]"},
            {"type": "plain", "text": "画好了。"},
            {"type": "image", "path": str(image_path)},
        ],
    }
    results = [
        result
        async for result in wrapper.call(
            ContextWrapper(
                context=SimpleNamespace(event=event, context=env.context),
                tool_call_timeout=10,
            ),
            **arguments,
        )
    ]
    assert results == [None] and event.get_extra("responses_image_delivered")
    target = adapter.bot.send_group_msg if group else adapter.bot.send_private_msg
    other = adapter.bot.send_private_msg if group else adapter.bot.send_group_msg
    target.assert_awaited_once()
    other.assert_not_awaited()
    payload = target.await_args.kwargs
    assert payload["group_id" if group else "user_id"] == (778899 if group else 10001)
    assert [part for part in payload["message"] if part["type"] == "at"] == (
        [{"type": "at", "data": {"qq": "10001"}}] if group else []
    )
    text = "".join(
        part["data"]["text"] for part in payload["message"] if part["type"] == "text"
    )
    assert text.strip() == "画好了。"
    images = [part for part in payload["message"] if part["type"] == "image"]
    assert len(images) == 1 and images[0]["data"]["file"].startswith("base64://")
    assert (
        base64.b64decode(images[0]["data"]["file"].removeprefix("base64://"))
        == image_path.read_bytes()
    )
    env.context.get_platform_inst.assert_not_called()
    env.context.send_message.assert_awaited_once()
    env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_image_does_not_send_a_standalone_mention_or_caption(control_env):
    event = FakeEvent()
    event.unified_msg_origin = "qq:GroupMessage:room"
    event.set_extra("responses_image_ready", True)
    event.set_extra("responses_image_requester", "original-user")
    args = {"messages": [{"type": "plain", "text": "画好了"}]}
    await control_env.plugin.restrict_image_delivery(
        event, SimpleNamespace(name="send_message_to_user"), args
    )
    assert args["messages"] == []


@pytest.mark.asyncio
async def test_ordinary_messages_are_not_modified_by_image_delivery_hooks(control_env):
    event = FakeEvent()
    args = {
        "session": "other",
        "messages": [{"type": "plain", "text": "ordinary chat"}],
    }
    original = copy.deepcopy(args)
    await control_env.plugin.restrict_image_delivery(
        event, SimpleNamespace(name="send_message_to_user"), args
    )
    assert args == original


@pytest.mark.asyncio
async def test_final_image_chain_keeps_caption_and_mentions_original_requester(
    control_env,
):
    event = FakeEvent()
    event.unified_msg_origin = "qq:GroupMessage:room"
    event.set_extra("responses_image_ready", True)
    event.set_extra("responses_image_requester", "original-user")
    image = Comp.Image.fromURL("https://example.invalid/generated.png")
    final = MessageEventResult(
        chain=[Comp.At(qq="wrong-person"), Comp.Plain("画好了。"), image]
    )
    event.get_result = lambda: final
    await control_env.plugin.image_delivery_reply(event)
    assert [str(part.qq) for part in final.chain if isinstance(part, Comp.At)] == [
        "original-user"
    ]
    assert final.get_plain_text() == "画好了。"
    assert final.chain[-1] is image


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["qq_official", "qq_official_webhook"])
@pytest.mark.parametrize("caption", [False, True])
async def test_final_qq_image_chain_uses_a_tag_free_fallback(
    control_env, platform, caption
):
    event = FakeEvent()
    event.platform_name = platform
    event.unified_msg_origin = "qq:GroupMessage:room"
    event.set_extra("responses_image_ready", True)
    event.set_extra("responses_image_requester", "original-openid")
    image = Comp.Image.fromURL("https://example.invalid/generated.png")
    final = MessageEventResult(
        chain=[
            Comp.At(qq="wrong-person"),
            Comp.Plain('<qqbot-at-user id="wrong-person" />'),
            Comp.Plain("画好了。" if caption else ""),
            image,
        ]
    )
    event.get_result = lambda: final
    await control_env.plugin.image_delivery_reply(event)
    assert final.chain[-1] is image
    assert not any(isinstance(part, Comp.At) for part in final.chain)
    assert final.get_plain_text() == ("画好了。" if caption else "图片已生成。")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    ["success", "error", "wrong_session", "no_receipt", "exception", "ordinary"],
)
async def test_delivery_wrapper_only_finishes_confirmed_image_sends(
    control_env, outcome
):
    import httpx
    from astrbot.core.agent.run_context import ContextWrapper
    from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
    from astrbot.core.agent.tool import FunctionTool, ToolSet
    from astrbot.core.astr_agent_hooks import EmptyAgentHooks
    from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
    from astrbot.core.provider.entities import LLMResponse

    env = control_env
    env.provider.provider_config = {"id": "offline-main"}
    env.provider.model_name = "offline-main"
    env.provider.client = SimpleNamespace(
        base_url=httpx.URL("https://example.invalid/v1")
    )
    event = FakeEvent()
    event.clear_result = Mock()
    event.get_result = Mock(return_value=None)
    event.set_extra("responses_image_ready", outcome != "ordinary")
    receipt = f"Message sent to session {event.unified_msg_origin}"
    if outcome == "error":
        receipt = CallToolResult(
            isError=True, content=[TextContent(type="text", text=receipt)]
        )
    elif outcome == "wrong_session":
        receipt = "Message sent to session qq:FriendMessage:other"
    elif outcome == "no_receipt":
        receipt = None
    sending = FunctionTool(
        name="send_message_to_user",
        description="Offline sender",
        parameters={"type": "object", "properties": {"messages": {"type": "array"}}},
        handler=AsyncMock(return_value=receipt),
    )
    if outcome == "exception":
        sending.handler.side_effect = RuntimeError("Offline send failed")
    tools = ToolSet([env.tools.get_func("responses_draw_image"), sending])
    request = ProviderRequest(prompt="Send image", func_tool=tools)
    await env.plugin.prepare_image_request(event, request)
    wrapper = request.func_tool.get_tool("send_message_to_user")
    await env.plugin.prepare_image_request(event, request)
    assert wrapper is request.func_tool.get_tool("send_message_to_user")
    assert tools.get_tool("send_message_to_user") is sending
    assert wrapper.original is sending
    first = LLMResponse(
        role="assistant",
        tools_call_name=["send_message_to_user"],
        tools_call_args=[{"messages": [{"type": "image", "path": "/generated.png"}]}],
        tools_call_ids=["send"],
    )
    # A send failure must permit a normal explanation; success never consumes it.
    env.provider.text_chat = AsyncMock(
        side_effect=[
            first,
            LLMResponse(role="assistant", completion_text="Delivery status explained."),
        ]
    )
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=env.provider,
        request=request,
        run_context=ContextWrapper(
            context=SimpleNamespace(event=event), tool_call_timeout=10
        ),
        tool_executor=FunctionToolExecutor(),
        agent_hooks=EmptyAgentHooks(),
        streaming=False,
    )
    async for _ in runner.step_until_done(3):
        pass
    assert runner.state.name == "DONE" and not runner.was_aborted()
    assert env.provider.text_chat.await_count == (1 if outcome == "success" else 2)
    assert bool(event.get_extra("responses_image_delivered")) == (outcome == "success")
    sending.handler.assert_awaited_once()
    if outcome == "success":
        # A duplicate send in the same model tool batch cannot resend the image.
        results = [
            result
            async for result in wrapper.call(
                runner.run_context,
                messages=[{"type": "image", "path": "/generated.png"}],
            )
        ]
        assert results == [None]
        sending.handler.assert_awaited_once()


@pytest.fixture
def qq_wire_sender():
    """Load actual QQ formatting methods without requiring its optional SDK.

    Returns:
        Adapter using real core payload construction and a mocked network client.
    """
    import ast
    import logging
    import os
    import random
    from pathlib import Path
    from typing import Any, cast

    import astrbot
    from astrbot.core.message.message_event_result import MessageChain
    from astrbot.core.platform.message_session import MessageSession
    from astrbot.core.platform.message_type import MessageType
    from astrbot.core.utils.media_utils import (
        MediaResolver,
        file_uri_to_path,
        is_file_uri,
    )

    namespace = dict(vars(Comp))
    namespace.update(
        Any=Any,
        os=os,
        random=random,
        MessageChain=MessageChain,
        SimpleNamespace=SimpleNamespace,
        MessageSesion=MessageSession,
        MessageType=MessageType,
        MediaResolver=MediaResolver,
        file_uri_to_path=file_uri_to_path,
        is_file_uri=is_file_uri,
        MarkdownPayload=dict,
        logger=logging.getLogger("qq-wire-test"),
        Platform=SimpleNamespace(send_by_session=AsyncMock()),
        _QQOFFICIAL_SEND_API_ERRORS=(RuntimeError,),
        botpy=SimpleNamespace(errors=SimpleNamespace(ServerError=RuntimeError)),
        cast=cast,
    )
    core = Path(astrbot.__file__).parent / "core/platform/sources/qqofficial"
    for file_name, class_name, methods in (
        (
            "qqofficial_message_event.py",
            "QQOfficialMessageEvent",
            {
                "_split_message_chain_by_media",
                "_parse_to_qqofficial",
                "_send_with_markdown_fallback",
            },
        ),
        (
            "qqofficial_platform_adapter.py",
            "QQOfficialPlatformAdapter",
            {
                "_send_by_session_common",
                "remember_session_message_id",
                "_extract_message_id",
            },
        ),
    ):
        source = ast.parse((core / file_name).read_text(encoding="utf-8"))
        cls = next(
            node
            for node in source.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        cls.body = [
            node
            for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in methods
        ]
        cls.bases = []
        cls.decorator_list = []
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                cls,
            ],
            type_ignores=[],
        )
        exec(
            compile(ast.fix_missing_locations(module), str(core / file_name), "exec"),
            namespace,
        )
    message_event = namespace["QQOfficialMessageEvent"]
    message_event._parse_keyboard = staticmethod(lambda chain: None)
    message_event.IMAGE_FILE_TYPE = 1
    message_event.MARKDOWN_NOT_ALLOWED_ERROR = "不允许发送原生 markdown"
    message_event.upload_group_and_c2c_image = AsyncMock(
        return_value={"file_info": "offline-upload"}
    )
    adapter = namespace["QQOfficialPlatformAdapter"]()
    adapter.client = SimpleNamespace(
        api=SimpleNamespace(
            post_group_message=AsyncMock(return_value={"id": "sent-message"}),
            post_message=AsyncMock(return_value={"id": "sent-message"}),
        )
    )
    adapter._session_scene = {"room": "group"}
    adapter._session_last_message_id = {"room": "received-message"}
    adapter._allow_group_proactive_send = False
    adapter.use_markdown_default = False
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["qq_official", "qq_official_webhook"])
@pytest.mark.parametrize("caption", [False, True])
@pytest.mark.parametrize("notice_fails", [False, True])
async def test_qq_image_and_mention_use_separate_actual_core_payloads(
    control_env, qq_wire_sender, platform, caption, notice_fails, tmp_path
):
    from astrbot.core.agent.run_context import ContextWrapper
    from astrbot.core.agent.tool import FunctionTool
    from astrbot.core.message.message_event_result import MessageChain
    from astrbot.core.platform.message_session import MessageSession

    env = control_env
    event = FakeEvent()
    event.unified_msg_origin = "qq:GroupMessage:room"
    event.platform_name = platform
    event.set_extra("responses_image_ready", True)
    event.set_extra("responses_image_requester", "original-openid")
    event.clear_result = Mock()
    event.get_result = Mock(return_value=None)
    adapter = qq_wire_sender
    adapter.use_markdown_default = True
    env.context.get_platform_inst = Mock(return_value=adapter)
    payloads = []

    async def post_group_message(**payload):
        """Capture the actual wire payload while replacing the QQ network call.

        Args:
            **payload: QQ send request from the real connector method.

        Returns:
            Offline API receipt.

        Raises:
            RuntimeError: Simulated notification failure after image success.
        """
        payloads.append(payload)
        if notice_fails and payload.get("msg_type") == 2:
            raise RuntimeError("Offline Markdown notification failed")
        return {"id": "sent-message"}

    adapter.client.api.post_group_message.side_effect = post_group_message

    async def send_message(session, chain):
        """Run real QQ payload construction without a live platform connection.

        Args:
            session: Original conversation identifier.
            chain: Plugin or tool message chain.

        Returns:
            True after the connector completes.
        """
        await adapter._send_by_session_common(
            MessageSession.from_str(str(session)), chain
        )
        return True

    env.context.send_message = AsyncMock(side_effect=send_message)

    async def deliver(message_event, messages):
        """Send the image through real connector methods for this delivery test.

        Args:
            message_event: Current execution event.
            messages: Filtered image-only tool arguments.

        Returns:
            The normal successful send-tool receipt.
        """
        assert all(part["type"] == "image" for part in messages)
        await env.context.send_message(
            message_event.unified_msg_origin,
            MessageChain(
                chain=[Comp.Image.fromFileSystem(part["path"]) for part in messages]
            ),
        )
        return f"Message sent to session {message_event.unified_msg_origin}"

    original = FunctionTool(
        name="send_message_to_user",
        description="Offline image sender",
        parameters={"type": "object", "properties": {"messages": {"type": "array"}}},
        handler=deliver,
    )
    wrapper = env.module.delivery.ImageDeliveryTool(original, env.plugin)
    messages = [{"type": "image", "path": str(tmp_path / "generated.png")}]
    if caption:
        messages.insert(0, {"type": "plain", "text": "画好了。"})
    stream = wrapper.call(
        ContextWrapper(context=SimpleNamespace(event=event), tool_call_timeout=10),
        messages=messages,
    )
    assert await anext(stream) is None
    assert event.get_extra("responses_image_delivered")
    assert [payload["msg_type"] for payload in payloads] == [7]
    # The main runner receives normal completion before the optional notification.
    assert [result async for result in stream] == []
    assert [payload["msg_type"] for payload in payloads] == [7, 2]
    assert not payloads[0].get("content") and "markdown" not in payloads[0]
    assert payloads[1]["markdown"][
        "content"
    ] == '<qqbot-at-user id="original-openid" /> ' + (
        "画好了。" if caption else "图片已生成。"
    )
    assert "media" not in payloads[1] and "content" not in payloads[1]
    assert all(payload["group_openid"] == "room" for payload in payloads)
    assert env.context.send_message.await_count == 1
    env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["qq_official", "qq_official_webhook"])
@pytest.mark.parametrize(
    "case",
    [
        "disabled",
        "rejected_cn",
        "rejected_content",
        "rejected_code",
        "rejected_render",
        "unknown_scene",
        "missing_sdk",
        "channel",
        "guild",
        "timeout",
        "rate_limit",
    ],
)
async def test_qq_notifications_fall_back_without_raw_tags(
    control_env, qq_wire_sender, platform, case
):
    from astrbot.core.platform.message_session import MessageSession

    env = control_env
    adapter = qq_wire_sender
    adapter.use_markdown_default = case != "disabled"
    # Group-member isolation must not change the native group destination.
    event = FakeEvent()
    event.unified_msg_origin = "qq:GroupMessage:12345_room"
    event.platform_name = platform
    env.context.get_platform_inst = Mock(return_value=adapter)
    channel = case in {"channel", "guild"}
    if channel:
        adapter._session_scene["room"] = case
    elif case == "unknown_scene":
        adapter._session_scene.clear()
    payloads = []
    errors = {
        "rejected_cn": RuntimeError("不允许发送原生 markdown"),
        "rejected_content": RuntimeError("不允许发送 markdown content"),
        "rejected_code": RuntimeError("code: 50056"),
        "rejected_render": RuntimeError("code: 40034090"),
        "timeout": TimeoutError("Uncertain send status"),
        "rate_limit": RuntimeError("22009: message limit exceeded"),
    }

    async def capture(**payload):
        """Reject Markdown while preserving every actual attempted payload.

        Args:
            **payload: Actual QQ SDK call arguments.

        Returns:
            A successful plain-text receipt.

        Raises:
            Exception: The selected simulated API failure.
        """
        payloads.append(copy.deepcopy(payload))
        if payload.get("msg_type") == 2 and case in errors:
            raise errors[case]
        return {"id": "sent-message"}

    adapter.client.api.post_group_message.side_effect = capture
    adapter.client.api.post_message.side_effect = capture

    async def send_message(session, chain):
        """Exercise core's connector fallback with no network activity.

        Args:
            session: Original conversation identifier.
            chain: Plugin fallback message.

        Returns:
            True when the connector completes.
        """
        await adapter._send_by_session_common(
            MessageSession.from_str(str(session)), chain
        )
        return True

    env.context.send_message = AsyncMock(side_effect=send_message)
    if case == "missing_sdk":
        # Keep the context sender functional while the direct API is unavailable.
        env.context.get_platform_inst.return_value = SimpleNamespace(
            _session_scene={"room": "group"},
            use_markdown_default=True,
        )
    wrapper = env.module.delivery.ImageDeliveryTool(
        SimpleNamespace(
            name="send_message_to_user", description="", parameters={}, active=True
        ),
        env.plugin,
    )
    notice = [
        {"type": "mention_user", "mention_user_id": "original-openid"},
        {"type": "plain", "text": "画好了。"},
    ]
    if case in {"timeout", "rate_limit"}:
        with pytest.raises(type(errors[case])):
            await wrapper._send_qq_notification(event, notice)
        assert len(payloads) == 1 and payloads[0]["msg_type"] == 2
        env.context.send_message.assert_not_awaited()
        return
    await wrapper._send_qq_notification(event, notice)
    env.context.get_platform_inst.assert_called_once_with("qq")
    assert len(payloads) == (2 if case.startswith("rejected") else 1)
    if case.startswith("rejected"):
        assert payloads[0]["msg_type"] == 2
        assert payloads[0]["msg_id"] == payloads[1]["msg_id"] == "received-message"
    plain = payloads[-1]
    assert "markdown" not in plain and plain.get("msg_type", 0) == 0
    assert plain.get("group_openid", plain.get("channel_id")) == "room"
    assert (
        plain["content"]
        == ('<qqbot-at-user id="original-openid" />' if channel else "") + "画好了。"
    )
    if not channel:
        assert not any(
            marker in plain["content"]
            for marker in ("<qqbot-", "<@", "[CQ:", "original-openid")
        )
    assert env.context.send_message.await_args.args[1].use_markdown_ is False
    env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", ["qq_official", "qq_official_webhook"])
@pytest.mark.parametrize("caption", [False, True])
@pytest.mark.parametrize(
    "outcome", ["success", "transport_error", "unavailable", "file_rejected"]
)
async def test_qq_without_markdown_combines_caption_and_image_once(
    control_env, qq_wire_sender, monkeypatch, tmp_path, platform, caption, outcome
):
    from astrbot.core.agent.run_context import ContextWrapper
    from astrbot.core.platform.message_session import MessageSession
    from astrbot.core.tools.message_tools import SendMessageToUserTool

    env = control_env
    adapter = qq_wire_sender
    event = FakeEvent()
    event.unified_msg_origin = "qq:GroupMessage:12345_room"
    event.platform_name = platform
    event.clear_result = Mock()
    event.get_result = Mock(return_value=None)
    event.set_extra("responses_image_ready", True)
    event.set_extra("responses_image_requester", "original-openid")
    env.context.get_platform_inst = Mock(return_value=adapter)

    async def send_message(session, chain):
        """Exercise the actual connector payload through the core sending tool.

        Args:
            session: Trusted current conversation selected by the sending tool.
            chain: Generated image with a sanitized plain caption.

        Returns:
            Whether the platform exists, matching Context.send_message.
        """
        if outcome == "unavailable":
            return False
        await adapter._send_by_session_common(
            MessageSession.from_str(str(session)), chain
        )
        return True

    env.context.send_message = AsyncMock(side_effect=send_message)
    if outcome == "transport_error":
        adapter.client.api.post_group_message.side_effect = TimeoutError(
            "Send timed out"
        )
    original = SendMessageToUserTool()
    image_path = tmp_path / "generated.png"
    resolver = AsyncMock(return_value=(str(image_path), False))
    if outcome == "file_rejected":
        resolver.side_effect = PermissionError("File send is restricted")
    monkeypatch.setattr(original, "_resolve_path_from_sandbox", resolver)
    wrapper = env.module.delivery.ImageDeliveryTool(original, env.plugin)
    messages = [
        {"type": "mention_user", "mention_user_id": "all"},
        {"type": "plain", "text": '<qqbot-at-user id="wrong-user" />'},
        {"type": "plain", "text": "source https://reference.example"},
        *([{"type": "plain", "text": "画好了。"}] if caption else []),
        {"type": "image", "path": str(image_path)},
    ]
    results = [
        result
        async for result in wrapper.call(
            ContextWrapper(
                context=SimpleNamespace(event=event, context=env.context),
                tool_call_timeout=10,
            ),
            session="qq:GroupMessage:unrelated-room",
            messages=messages,
        )
    ]
    resolver.assert_awaited_once()
    if outcome == "success":
        assert results == [None]
        assert event.get_extra("responses_image_delivered") is True
        event.clear_result.assert_called_once()
    else:
        assert not event.get_extra("responses_image_delivered")
        assert results and all(isinstance(result, CallToolResult) for result in results)
        assert any(
            isinstance(part, TextContent) and "error:" in part.text
            for result in results
            for part in result.content
        )
        event.clear_result.assert_not_called()
    assert env.context.send_message.await_count == (outcome != "file_rejected")
    if outcome in {"success", "transport_error"}:
        adapter.client.api.post_group_message.assert_awaited_once()
        payload = adapter.client.api.post_group_message.await_args.kwargs
        assert payload["group_openid"] == "room" and payload["msg_type"] == 7
        assert payload["media"] == {"file_info": "offline-upload"}
        assert payload["content"] == ("画好了。" if caption else "图片已生成。")
        assert "markdown" not in payload
    else:
        adapter.client.api.post_group_message.assert_not_awaited()
    adapter.client.api.post_message.assert_not_awaited()
    env.provider.generate_image.assert_not_awaited()
