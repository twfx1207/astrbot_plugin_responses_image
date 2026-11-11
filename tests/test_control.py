"""Exercise the controlled entry point with real AstrBot messages and tool schemas."""

import asyncio
import copy
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio
from astrbot.core.message import components as Comp
from astrbot.core.message.message_event_result import MessageChain, MessageEventResult
from astrbot.core.platform.astr_message_event import AstrMessageEvent
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from mcp.types import CallToolResult, ImageContent
from PIL import Image as PillowImage


class DrawingTransportMock(AsyncMock):
    """Simulate the post-validation drawing transport boundary."""

    async def __call__(self, *args, before_request=None, **kwargs):
        """Commit quota immediately before the simulated upstream call.

        Args:
            *args: Drawing inputs.
            before_request: Optional quota commit callback.
            **kwargs: Request options captured by the transport mock.

        Returns:
            Mocked image response.
        """
        if before_request is not None:
            before_request()
        return await super().__call__(*args, **kwargs)


class FakeEvent:
    """Keep current messages separate from unrelated conversation data."""

    plain_result = AstrMessageEvent.plain_result
    chain_result = AstrMessageEvent.chain_result

    def __init__(self, text="Draw a blue bird.", messages=None, *, admin=False):
        """Prepare an event with real AstrBot message components.

        Args:
            text: Original incoming user text.
            messages: Explicit current attachments and quoted messages.
            admin: Whether the sender has administrator permission.
        """
        self.text = text
        self.messages = messages or []
        self.admin = admin
        self._extras = {}
        self.unified_msg_origin = "qq:FriendMessage:test-user"
        self.history_images = [Comp.Image.fromBase64("unrelated-old-image")]

    def get_messages(self):
        """Return current message components.

        Returns:
            Current message components, without history.
        """
        return self.messages

    def get_message_str(self):
        """Return the original incoming text.

        Returns:
            Current message text.
        """
        return self.text

    def set_extra(self, key, value):
        """Save event-local metadata.

        Args:
            key: Metadata key.
            value: Metadata value.
        """
        self._extras[key] = value

    def get_extra(self, key, default=None):
        """Read event-local metadata.

        Args:
            key: Metadata key.
            default: Fallback when no value exists.

        Returns:
            Stored metadata or the default.
        """
        return self._extras.get(key, default)

    def is_admin(self):
        """Report administrator permission.

        Returns:
            Whether this event was created for an administrator.
        """
        return self.admin

    def get_platform_id(self):
        """Return the connection identity.

        Returns:
            Platform identifier used to scope account IDs.
        """
        return getattr(self, "platform_id", "qq")

    def get_platform_name(self):
        """Return the adapter type, independently of the configured connection ID.

        Returns:
            Adapter name used to select platform-specific delivery behavior.
        """
        return getattr(self, "platform_name", "aiocqhttp")

    def get_sender_id(self):
        """Return the submitting account.

        Returns:
            Stable account ID independent of the chat window.
        """
        return getattr(self, "sender_id", "test-user")

    def get_message_type(self):
        """Return the window type encoded in the fixture session.

        Returns:
            AstrBot's group, private or other message type.
        """
        from astrbot.core.platform.message_type import MessageType

        return MessageType(self.unified_msg_origin.split(":", 2)[1])


@pytest.fixture
def control_env(monkeypatch, tmp_path):
    """Import the plugin against fresh real tool and handler registries.

    Args:
        monkeypatch: Fixture that restores registration state.
        tmp_path: Isolated plugin data directory.

    Yields:
        Plugin instance, mocked worker and real registration metadata.
    """
    from astrbot.core.provider.func_tool_manager import FunctionToolManager
    from astrbot.core.star.star_handler import StarHandlerRegistry

    registration = importlib.import_module("astrbot.core.star.register.star_handler")
    star_base = importlib.import_module("astrbot.core.star.base")
    tool_manager = FunctionToolManager()
    monkeypatch.setattr(registration, "llm_tools", tool_manager)
    monkeypatch.setattr(registration, "star_handlers_registry", StarHandlerRegistry())
    monkeypatch.setattr(star_base, "star_map", {})
    monkeypatch.setattr(star_base, "star_registry", [])
    package_name = "responses_image_control_under_test"
    package = ModuleType(package_name)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    monkeypatch.setitem(sys.modules, package_name, package)
    module_name = f"{package_name}.main"
    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).resolve().parents[1] / "main.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    # Control tests mock image operations while retaining real provider types.
    monkeypatch.setattr(module, "ResponsesImageClient", lambda provider: provider)
    provider = object.__new__(module.ProviderOpenAIResponses)
    generated_path = tmp_path / "generated.png"
    PillowImage.new("RGB", (2, 2), "blue").save(generated_path)
    provider.generate_image = DrawingTransportMock(
        return_value=LLMResponse(
            "assistant",
            result_chain=MessageChain(
                chain=[Comp.Image.fromFileSystem(generated_path)]
            ),
        )
    )
    provider.search_reference_images = AsyncMock(return_value=[])
    provider.verify_reference_image = AsyncMock(return_value=False)
    context = SimpleNamespace(
        get_current_chat_provider_id=AsyncMock(return_value="current-worker"),
        get_provider_by_id=Mock(return_value=provider),
        get_config=Mock(return_value={"provider_settings": {}}),
    )
    plugin = module.ResponsesImagePlugin(context)
    plugin.library = module.CharacterLibrary(tmp_path / "library")
    plugin.quota = module.DrawingQuota(tmp_path / "quota.json", plugin.config)
    try:
        yield SimpleNamespace(
            module=module,
            plugin=plugin,
            provider=provider,
            context=context,
            tools=tool_manager,
            handlers=registration.star_handlers_registry,
        )
    finally:
        for name in list(sys.modules):
            if name.startswith(package_name + "."):
                sys.modules.pop(name, None)


@pytest.fixture
def current_image(tmp_path):
    """Create a real current-message attachment.

    Args:
        tmp_path: Isolated filesystem fixture.

    Returns:
        An image component pointing at a real local PNG.
    """
    path = tmp_path / "current.png"
    PillowImage.new("RGB", (2, 2), "red").save(path)
    return Comp.Image.fromFileSystem(path)


@pytest_asyncio.fixture
async def background_env(control_env):
    """Provide real queued wakeups and bounded cleanup for background tests.

    Args:
        control_env: Plugin instance with isolated drawing transport.

    Yields:
        The control environment with a queue, session and completion factory.
    """
    from astrbot.core.agent.tool import FunctionTool, ToolSet
    from astrbot.core.platform.platform_metadata import PlatformMetadata

    env = control_env
    env.plugin.config["background_generation"] = True
    env.queue = asyncio.Queue()
    env.meta = PlatformMetadata("qq_official", "Offline QQ", "qq")
    env.context.get_platform_inst = Mock(
        return_value=SimpleNamespace(meta=lambda: env.meta)
    )
    env.context.get_event_queue = Mock(return_value=env.queue)
    env.context.send_message = AsyncMock(return_value=True)
    env.context.conversation_manager = SimpleNamespace(
        get_curr_conversation_id=AsyncMock(return_value="conversation")
    )
    env.background_event = FakeEvent()
    env.background_event.platform_meta = env.meta
    env.background_event.set_extra("responses_image_conversation_id", "conversation")

    def completion_request(receipt, cid="conversation"):
        """Recreate the persisted submission and current completion request.

        Args:
            receipt: Actual string returned by the submission tool.
            cid: Conversation currently selected when the wakeup acquires its lock.

        Returns:
            A request containing the submission record and available tools.
        """
        return ProviderRequest(
            conversation=SimpleNamespace(cid=cid, token_usage=0),
            contexts=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "submitted_draw",
                            "type": "function",
                            "function": {
                                "name": "responses_draw_image",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "submitted_draw", "content": receipt},
            ],
            func_tool=ToolSet(
                [
                    env.tools.get_func("responses_draw_image"),
                    FunctionTool(
                        name="send_message_to_user",
                        description="Send reviewed images",
                        parameters={},
                    ),
                    FunctionTool(
                        name="unrelated_tool",
                        description="Excluded from completion",
                        parameters={},
                    ),
                ]
            ),
        )

    env.completion_request = completion_request
    try:
        yield env
    finally:
        await env.plugin.terminate()


def test_registers_tool_schema_without_model_supplied_prompt_or_paths(control_env):
    tool = control_env.tools.get_func("responses_draw_image")
    assert tool is not None
    properties = tool.parameters["properties"]
    assert set(properties) == {
        "operation",
        "character_names",
        "reference_ids",
        "context_ids",
        "search_references",
    }
    assert properties["reference_ids"] == {
        "type": "array",
        "description": properties["reference_ids"]["description"],
        "items": {"type": "string"},
    }
    for key in ("character_names", "context_ids"):
        assert properties[key]["type"] == "array"
        assert properties[key]["items"] == {"type": "string"}
    assert "current user" in tool.description
    assert properties["search_references"]["type"] == "boolean"


@pytest.mark.asyncio
async def test_hook_preserves_chat_history_and_captures_only_event_images(
    control_env, current_image
):
    event = FakeEvent("Draw the current request.", [current_image])
    history = [{"role": "user", "content": "OLD_CHARACTER_SUMMARY"}]
    req = ProviderRequest(
        prompt="REWRITTEN_PROMPT_WITH_OLD_STYLE",
        contexts=copy.deepcopy(history),
        image_urls=["old://image", "current://image"],
        system_prompt="Normal chat persona.",
    )
    await control_env.plugin.prepare_image_request(event, req)
    assert req.contexts == history
    assert req.image_urls == ["old://image", "current://image"]
    assert req.prompt == "REWRITTEN_PROMPT_WITH_OLD_STYLE"
    assert req.system_prompt.startswith("Normal chat persona.")
    candidates = event.get_extra("responses_image_references")
    assert len(candidates) == 1
    assert next(iter(candidates.values()))["image"] is current_image
    assert event.get_extra("responses_image_user_text") == "Draw the current request."
    assert "old://image" not in req.system_prompt


@pytest.mark.asyncio
async def test_generation_uses_snapshot_not_rewritten_request_or_history(control_env):
    event = FakeEvent("Draw the current requested bird.")
    req = ProviderRequest(prompt="OLD_STYLE_IN_REWRITTEN_PROMPT")
    await control_env.plugin.prepare_image_request(event, req)
    event.text = "LATER_MUTATED_MESSAGE_WITH_OLD_STYLE"
    result = await control_env.plugin.draw_image(event, "new", [], [])
    assert isinstance(result, CallToolResult)
    prompt = control_env.provider.generate_image.await_args.args[0]
    assert "Draw the current requested bird." in prompt
    assert "OLD_STYLE" not in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_id", ["", "explicit-worker"])
async def test_worker_selection_uses_configuration_or_current_model(
    control_env, configured_id
):
    control_env.plugin.config["worker_provider_id"] = configured_id
    result = await control_env.plugin.draw_image(FakeEvent(), "new", [], [])
    assert isinstance(result, CallToolResult)
    control_env.context.get_provider_by_id.assert_called_once_with(
        configured_id or "current-worker"
    )
    if configured_id:
        control_env.context.get_current_chat_provider_id.assert_not_awaited()
    else:
        control_env.context.get_current_chat_provider_id.assert_awaited_once()


@pytest.mark.asyncio
async def test_unsupported_worker_returns_actionable_error(control_env):
    control_env.context.get_provider_by_id.return_value = object()
    result = await control_env.plugin.draw_image(FakeEvent(), "new", [], [])
    assert isinstance(result, str)
    assert "worker_provider_id" in result
    control_env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_generic_worker_is_selected_without_requiring_plugin_adapter(
    control_env, monkeypatch
):
    env = control_env
    generic = object.__new__(env.module.ProviderOpenAIResponses)
    env.context.get_provider_by_id.return_value = generic
    binding = Mock(return_value=env.provider)
    monkeypatch.setattr(env.module, "ResponsesImageClient", binding)
    result = await env.plugin.draw_image(FakeEvent(), "new", [], [])
    assert isinstance(result, CallToolResult)
    assert all(call.args == (generic,) for call in binding.call_args_list)
    env.provider.generate_image.assert_awaited_once()


@pytest.mark.asyncio
async def test_new_image_does_not_implicitly_use_current_or_historical_pictures(
    control_env, current_image
):
    event = FakeEvent("Draw a new landscape.", [current_image])
    result = await control_env.plugin.draw_image(event, "new", [], [])
    assert isinstance(result, CallToolResult)
    prompt, image_paths = control_env.provider.generate_image.await_args.args
    assert image_paths == []
    assert "unrelated-old-image" not in prompt
    assert "current.png" not in prompt
    assert len(result.content) == 2
    assert isinstance(result.content[1], ImageContent)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "adapter", ["ProviderOpenAIResponses", "ProviderOpenAIOfficial"]
)
async def test_images_api_option_selects_existing_connection(
    control_env, monkeypatch, adapter
):
    env = control_env
    generic = object.__new__(getattr(env.module, adapter))
    env.context.get_provider_by_id.return_value = generic
    env.plugin.config.update({"generation_api": "images", "image_model": "pure-image"})
    binding = Mock(return_value=env.provider)
    monkeypatch.setattr(env.module, "ImagesAPIClient", binding)
    result = await env.plugin.draw_image(FakeEvent(), "new", [], [])
    assert isinstance(result, CallToolResult)
    assert any(isinstance(part, ImageContent) for part in result.content)
    binding.assert_called_once_with(generic)
    assert env.provider.generate_image.await_args.kwargs == {
        "operation": "new",
        "image_model": "pure-image",
    }
    env.provider.search_reference_images.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_generation_api_does_not_call_worker(control_env):
    control_env.plugin.config["generation_api"] = "automatic"
    result = await control_env.plugin.draw_image(FakeEvent(), "new", [], [])
    assert "绘图接口配置无效" in result
    control_env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_character_without_reference_draws_from_name(
    control_env, current_image
):
    event = FakeEvent("Draw Sherry.", [current_image])
    result = await control_env.plugin.draw_image(event, "new", ["Sherry"], [])
    assert isinstance(result, CallToolResult)
    assert control_env.provider.generate_image.await_args.args[1] == []


@pytest.mark.asyncio
async def test_known_character_uses_registered_image_and_canonical_name(
    control_env, current_image
):
    profile = await control_env.plugin.library.save(
        "Work Character", ["Alias"], current_image
    )
    event = FakeEvent("Draw Alias smiling.")
    result = await control_env.plugin.draw_image(event, "new", ["alias"], [])
    assert isinstance(result, CallToolResult)
    prompt, paths = control_env.provider.generate_image.await_args.args
    assert paths == [profile["image_path"]]
    assert '"characters": ["Work Character"]' in prompt
    assert '"reference_source": "administrator_registered_character"' in prompt


@pytest.mark.asyncio
async def test_unknown_character_accepts_an_explicit_current_reference(
    control_env, current_image
):
    event = FakeEvent(
        "Draw Unregistered Character smiling using the second image.",
        [Comp.Image.fromBase64("unselected-image"), current_image],
    )
    refs = control_env.module.collect_event_images(event)
    selected_id = list(refs)[1]
    result = await control_env.plugin.draw_image(
        event, "new", ["Unregistered Character"], [selected_id]
    )
    assert isinstance(result, CallToolResult)
    assert control_env.provider.generate_image.await_args.args[1] == [
        await current_image.convert_to_file_path()
    ]
    prompt = control_env.provider.generate_image.await_args.args[0]
    task_data = json.loads(prompt.rsplit("\n", 1)[1])
    assert task_data["selected_references"] == [
        {
            "id": selected_id,
            "source": "attachment",
            "original_index": 2,
            "input_index": 1,
        }
    ]


@pytest.mark.asyncio
async def test_reference_from_another_event_is_rejected(control_env, current_image):
    previous = FakeEvent(messages=[current_image])
    stale_ids = list(control_env.module.collect_event_images(previous))
    current = FakeEvent(messages=[current_image])
    result = await control_env.plugin.draw_image(current, "new", [], stale_ids)
    assert "不属于本次可用候选" in result
    control_env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("character_name", ["", "Character"])
async def test_edit_requires_an_available_selected_target(control_env, character_name):
    result = await control_env.plugin.draw_image(
        FakeEvent(), "edit", [character_name] if character_name else [], []
    )
    assert "编辑图片须选择" in result
    control_env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_edit_uses_explicit_quoted_target(control_env, current_image):
    event = FakeEvent(
        "Make the quoted image brighter.",
        [Comp.Reply(id="explicit-quote", chain=[current_image])],
    )
    ids = list(control_env.module.collect_event_images(event))
    result = await control_env.plugin.draw_image(event, "edit", [], ids)
    assert isinstance(result, CallToolResult)
    assert control_env.provider.generate_image.await_args.kwargs == {
        "operation": "edit"
    }
    assert len(control_env.provider.generate_image.await_args.args[1]) == 1


@pytest.mark.asyncio
async def test_registered_character_cannot_mix_with_current_reference(
    control_env, current_image
):
    await control_env.plugin.library.save("Character", [], current_image)
    event = FakeEvent("Draw Character.", [current_image])
    ids = list(control_env.module.collect_event_images(event))
    result = await control_env.plugin.draw_image(event, "new", ["Character"], ids)
    assert "勿混用" in result
    control_env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_known_name_in_user_request_selects_profile_without_tool_character(
    control_env, current_image
):
    profile = await control_env.plugin.library.save("雪莉", ["Sherry"], current_image)
    result = await control_env.plugin.draw_image(
        FakeEvent("画一张雪莉。"), "new", [], []
    )
    assert isinstance(result, CallToolResult)
    prompt, paths = control_env.provider.generate_image.await_args.args
    assert paths == [profile["image_path"]]
    assert '"characters": ["雪莉"]' in prompt


@pytest.mark.asyncio
async def test_tool_cannot_substitute_different_registered_character(
    control_env, current_image
):
    await control_env.plugin.library.save("雪莉", [], current_image)
    await control_env.plugin.library.save("爱乃", [], current_image)
    result = await control_env.plugin.draw_image(
        FakeEvent("画一张雪莉。"), "new", ["爱乃"], []
    )
    assert isinstance(result, str)
    control_env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_name_only_present_in_history_cannot_select_registered_character(
    control_env, current_image
):
    await control_env.plugin.library.save("爱乃", [], current_image)
    event = FakeEvent("画一只鸟。")
    await control_env.plugin.prepare_image_request(
        event, ProviderRequest(contexts=[{"role": "user", "content": "画爱乃。"}])
    )
    result = await control_env.plugin.draw_image(event, "new", ["爱乃"], [])
    assert isinstance(result, str)
    control_env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_excluded_current_character_is_not_forced_into_generation(
    control_env, current_image
):
    await control_env.plugin.library.save("雪莉", [], current_image)
    await control_env.plugin.library.save("爱乃", [], current_image)
    result = await control_env.plugin.draw_image(
        FakeEvent("只画雪莉，不要爱乃。"), "new", ["雪莉"], []
    )
    assert isinstance(result, CallToolResult)
    prompt, paths = control_env.provider.generate_image.await_args.args
    data = json.loads(prompt.rsplit("\n", 1)[1])
    assert data["characters"] == ["雪莉"]
    assert len(paths) == 1
    assert [item["character"] for item in data["character_references"]] == ["雪莉"]


@pytest.mark.asyncio
async def test_unknown_tool_character_not_in_original_text_is_rejected(
    control_env, current_image
):
    event = FakeEvent("Draw a bird using this composition.", [current_image])
    refs = list(control_env.module.collect_event_images(event))
    result = await control_env.plugin.draw_image(
        event, "new", ["A character from history"], refs
    )
    assert isinstance(result, str)
    control_env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_calls_on_one_event_make_only_one_worker_request(control_env):
    event = FakeEvent()
    started = asyncio.Event()
    release = asyncio.Event()
    expected_result = control_env.provider.generate_image.return_value

    async def generate(*args, **kwargs):
        started.set()
        await release.wait()
        return expected_result

    control_env.provider.generate_image.side_effect = generate
    first = asyncio.create_task(control_env.plugin.draw_image(event, "new", [], []))
    await started.wait()
    second = asyncio.create_task(control_env.plugin.draw_image(event, "new", [], []))
    release.set()
    results = await asyncio.gather(first, second)
    assert sum(isinstance(result, CallToolResult) for result in results) == 1
    assert any(isinstance(result, str) and "已经发起过" in result for result in results)
    control_env.provider.generate_image.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_worker_attempt_is_not_retried_for_same_event(control_env):
    event = FakeEvent()
    control_env.provider.generate_image.side_effect = TimeoutError("private body")
    first = await control_env.plugin.draw_image(event, "new", [], [])
    second = await control_env.plugin.draw_image(event, "new", [], [])
    assert "TimeoutError" in first
    assert "private body" not in first
    assert "已经发起过" in second
    control_env.provider.generate_image.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_worker_keeps_attempt_marker_and_releases_event_lock(
    control_env,
):
    event = FakeEvent()
    started = asyncio.Event()
    pending = asyncio.Event()

    async def generate(*args, **kwargs):
        started.set()
        await pending.wait()

    control_env.provider.generate_image.side_effect = generate
    task = asyncio.create_task(control_env.plugin.draw_image(event, "new", [], []))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert event.get_extra("responses_image_attempted") is True
    assert not event.get_extra("responses_image_lock").locked()
    result = await asyncio.wait_for(
        control_env.plugin.draw_image(event, "new", [], []), timeout=2
    )
    assert "已经发起过" in result
    control_env.provider.generate_image.assert_awaited_once()


@pytest.mark.asyncio
async def test_separate_events_each_get_their_own_worker_attempt(control_env):
    results = await asyncio.gather(
        control_env.plugin.draw_image(FakeEvent(), "new", [], []),
        control_env.plugin.draw_image(FakeEvent(), "new", [], []),
    )
    assert all(isinstance(result, CallToolResult) for result in results)
    assert control_env.provider.generate_image.await_count == 2


@pytest.mark.asyncio
async def test_text_only_worker_result_is_not_treated_as_generated_image(control_env):
    control_env.provider.generate_image.return_value = LLMResponse(
        "assistant", result_chain=MessageChain(chain=[Comp.Plain("No image")])
    )
    result = await control_env.plugin.draw_image(FakeEvent(), "new", [], [])
    assert "没有返回完整图片" in result


@pytest.mark.asyncio
async def test_model_with_no_delivery_tool_does_not_start_generation(control_env):
    from astrbot.core.agent.tool import ToolSet

    event = FakeEvent()
    await control_env.plugin.prepare_image_request(
        event, ProviderRequest(func_tool=ToolSet())
    )
    result = await control_env.plugin.draw_image(event, "new", [], [])
    assert "send_message_to_user" in result
    control_env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_image_model_panel_value_reaches_only_generation(control_env):
    control_env.plugin.config["image_model"] = " chosen-image-model "
    result = await control_env.plugin.draw_image(FakeEvent(), "new", [], [])
    assert isinstance(result, CallToolResult)
    assert control_env.provider.generate_image.await_args.kwargs == {
        "operation": "new",
        "image_model": "chosen-image-model",
    }
    assert "NOT been sent" in result.content[0].text


@pytest.mark.native_wakeup
@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("group", [False, True])
async def test_real_agent_receives_pixels_before_delivery_and_persists_them(
    background_env, monkeypatch, tmp_path, streaming, background, group
):
    import base64

    import astrbot.core.astr_agent_hooks as agent_hooks_module
    import httpx
    from astrbot.core.agent.message import ImageURLPart
    from astrbot.core.agent.run_context import ContextWrapper
    from astrbot.core.agent.runners.tool_loop_agent_runner import ToolLoopAgentRunner
    from astrbot.core.agent.tool import FunctionTool, ToolSet
    from astrbot.core.agent.tool_image_cache import tool_image_cache
    from astrbot.core.astr_agent_tool_exec import FunctionToolExecutor
    from astrbot.core.pipeline.process_stage.method.agent_sub_stages.internal import (
        InternalAgentSubStage,
    )
    from astrbot.core.star.star_handler import EventType

    env = background_env

    async def dispatch_hook(message_event, event_type, *args):
        """Exercise main-agent hooks with this isolated plugin registry.

        Args:
            message_event: Actual execution event.
            event_type: AstrBot hook type.
            *args: Tool and execution arguments.
        """
        if event_type == EventType.OnUsingLLMToolEvent:
            await env.plugin.restrict_image_delivery(message_event, *args)

    monkeypatch.setattr(agent_hooks_module, "call_event_hook", dispatch_hook)
    env.plugin.config["background_generation"] = background
    monkeypatch.setattr(tool_image_cache, "_cache_dir", str(tmp_path / "tool-cache"))
    event = FakeEvent()
    event.sender_id = env.background_event.sender_id = "original-requester"
    if group:
        event.unified_msg_origin = env.background_event.unified_msg_origin = (
            "qq:GroupMessage:shared-group"
        )
    receipt = None
    job = None
    if background:
        receipt = await env.plugin.draw_image(env.background_event, "new", [], [])
        env.background_event.sender_id = "later-speaker"
        job = next(iter(env.plugin._jobs.values()))
        event = await asyncio.wait_for(env.queue.get(), timeout=2)
        event.text = env.background_event.text
    event.send = AsyncMock()
    # Connector formatting is exercised separately with the QQ adapter send path.
    event.get_platform_name = lambda: "aiocqhttp"
    event.set_result = Mock()
    event.get_result = Mock(return_value=None)
    event.clear_result = Mock()
    drawing = env.tools.get_func("responses_draw_image")
    drawing.handler = env.plugin.draw_image
    viewed_paths = []

    async def deliver(message_event, messages):
        """Deliver only after the simulated main model has received the pixels.

        Args:
            message_event: Current event used by the actual local tool executor.
            messages: Main model's explicit delivery arguments.

        Returns:
            A textual receipt that keeps the main loop running.
        """
        assert viewed_paths == [messages[-1]["path"]]
        assert messages[-1]["type"] == "image"
        assert [part for part in messages if part["type"] == "mention_user"] == (
            [{"type": "mention_user", "mention_user_id": "original-requester"}]
            if group
            else []
        )
        assert [part["text"] for part in messages if part["type"] == "plain"] == [
            "画好了，背景使用了蓝色。"
        ]
        chain = [Comp.At(qq="original-requester")] if group else []
        chain += [
            Comp.Plain("画好了，背景使用了蓝色。"),
            Comp.Image.fromFileSystem(messages[-1]["path"]),
        ]
        await message_event.send(MessageChain(chain=chain))
        return "Message sent to session " + message_event.unified_msg_origin

    sending = FunctionTool(
        name="send_message_to_user",
        description="Send reviewed images",
        parameters={
            "type": "object",
            "properties": {"messages": {"type": "array", "items": {"type": "object"}}},
        },
        handler=deliver,
    )
    req = ProviderRequest(prompt=event.text, func_tool=ToolSet([drawing, sending]))
    if background:
        completed = env.completion_request(receipt)
        req.contexts = completed.contexts
        req.conversation = completed.conversation
    await env.plugin.prepare_image_request(event, req)
    main_provider = object.__new__(env.module.ProviderOpenAIResponses)
    main_provider.provider_config = {"id": "main", "modalities": ["text", "image"]}
    main_provider.model_name = "main-vision-model"
    main_provider.client = SimpleNamespace(
        base_url=httpx.URL("https://example.invalid/v1")
    )
    calls = []

    async def respond(**kwargs):
        """Inspect actual requests between generation and sending.

        Args:
            **kwargs: Main model request assembled by AstrBot's real agent runner.

        Returns:
            The next controlled model response in the offline workflow.
        """
        calls.append(kwargs)
        if len(calls) == 1:
            return LLMResponse(
                role="assistant",
                tools_call_name=["responses_draw_image"],
                tools_call_args=[
                    {"operation": "new", "character_names": [], "reference_ids": []}
                ],
                tools_call_ids=["call_draw"],
            )
        if len(calls) == 2:
            event.send.assert_not_awaited()
            messages = [
                item if isinstance(item, dict) else item.model_dump()
                for item in kwargs["contexts"]
            ]
            images = [
                part["image_url"]
                for message in messages
                for part in (
                    message["content"]
                    if isinstance(message.get("content"), list)
                    else []
                )
                if part.get("type") == "image_url"
            ]
            assert len(images) == 1 and images[0]["url"].startswith(
                "data:image/png;base64,"
            )
            viewed_paths.append(images[0]["id"])
            assert (
                base64.b64decode(images[0]["url"].split(",", 1)[1])
                == (tmp_path / "generated.png").read_bytes()
            )
            assert Path(viewed_paths[0]).is_file()
            return LLMResponse(
                role="assistant",
                tools_call_name=["send_message_to_user"],
                tools_call_args=[
                    {
                        "messages": [
                            {
                                "type": "plain",
                                "text": "Source https://reference.example",
                            },
                            {"type": "mention_user", "mention_user_id": "wrong-person"},
                            {"type": "plain", "text": "画好了，背景使用了蓝色。"},
                            {"type": "image", "path": viewed_paths[0]},
                        ]
                    }
                ],
                tools_call_ids=["call_send"],
            )
        pytest.fail("Successful image delivery must not call the model again")

    async def stream_response(**kwargs):
        """Return the same offline response through AstrBot's streaming route.

        Args:
            **kwargs: Main model request captured by the shared responder.

        Yields:
            The complete response without simulating network chunks.
        """
        yield await respond(**kwargs)

    main_provider.text_chat = AsyncMock(side_effect=respond)
    main_provider.text_chat_stream = stream_response
    runner = ToolLoopAgentRunner()
    await runner.reset(
        provider=main_provider,
        request=req,
        run_context=ContextWrapper(
            context=SimpleNamespace(event=event), tool_call_timeout=60
        ),
        tool_executor=FunctionToolExecutor(),
        agent_hooks=agent_hooks_module.MainAgentHooks(),
        streaming=streaming,
    )
    async for _ in runner.step_until_done(4):
        pass
    assert len(calls) == 2
    assert runner.state.name == "DONE" and not runner.was_aborted()
    event.send.assert_awaited_once()
    assert event.get_extra("responses_image_delivered") is True
    event.set_result.assert_not_called()
    final = MessageEventResult().message("Stale final reply must not be sent again")
    event.get_result.return_value = final
    await env.plugin.image_delivery_reply(event)
    assert final.chain == []
    event.get_result.return_value = None
    env.provider.generate_image.assert_awaited_once()
    if background:
        await asyncio.wait_for(job["task"], timeout=2)
        assert env.plugin._jobs == {}
        env.plugin.config["background_generation"] = False
    assert any(
        isinstance(part, ImageURLPart)
        for message in runner.run_context.messages
        if isinstance(message.content, list)
        for part in message.content
    )
    conv_manager = SimpleNamespace(update_conversation=AsyncMock())
    req.conversation = SimpleNamespace(cid="conversation", token_usage=0)
    await InternalAgentSubStage._save_to_history(
        SimpleNamespace(conv_manager=conv_manager),
        event,
        req,
        runner.get_final_llm_resp(),
        runner.run_context.messages,
        None,
    )
    saved = conv_manager.update_conversation.await_args.kwargs["history"]
    assert (
        len(
            [
                message
                for message in saved
                if message.get("role") == "tool"
                and message.get("tool_call_id") == "call_send"
            ]
        )
        == 1
    )
    wire = main_provider._convert_chat_messages_to_response_input(saved)
    assert any(
        part.get("type") == "input_image"
        for message in wire
        if message.get("type") == "message" and isinstance(message.get("content"), list)
        for part in message["content"]
    )
    assert all(
        message.get("role") != "user" or not message.get("_no_save")
        for message in saved
    )

    # A new event can edit the persisted image even after the cache disappears.
    Path(viewed_paths[0]).unlink()
    edit_event = FakeEvent("Make the background of the image just generated darker.")
    edit_req = ProviderRequest(contexts=copy.deepcopy(saved), func_tool=req.func_tool)
    await env.plugin.prepare_image_request(edit_event, edit_req)
    edit_candidates = edit_event.get_extra("responses_image_generated_references")
    assert len(edit_candidates) == 1
    assert "Latest generated image edit candidates" in edit_req.system_prompt
    assert "data:image/" not in edit_req.system_prompt
    generated = env.provider.generate_image.return_value
    edited_inputs = []

    async def edit_image(prompt, images, **options):
        """Capture the exact approved edit pixels before temporary cleanup.

        Args:
            prompt: Isolated current edit request.
            images: Selected edit targets only.
            **options: Generation options including the edit action.

        Returns:
            The existing offline image result.
        """
        assert options["operation"] == "edit"
        assert len(images) == 1
        edited_inputs.append(Path(images[0]).read_bytes())
        assert edit_event.text in prompt and event.text not in prompt
        assert "selected_generated_edit_images" in prompt
        return generated

    env.provider.generate_image.side_effect = edit_image
    assert isinstance(
        await env.plugin.draw_image(edit_event, "edit", [], list(edit_candidates)),
        CallToolResult,
    )
    assert edited_inputs == [(tmp_path / "generated.png").read_bytes()]
    env.provider.search_reference_images.assert_not_awaited()
    fresh_event = FakeEvent("Draw something unrelated.")
    await env.plugin.prepare_image_request(fresh_event, ProviderRequest())
    assert fresh_event.get_extra("responses_image_generated_references") == {}
    assert "不属于本次可用候选" in await env.plugin.draw_image(
        fresh_event, "edit", [], list(edit_candidates)
    )
    assert "不属于本次可用候选" in await env.plugin.draw_image(
        edit_event, "new", [], list(edit_candidates)
    )


@pytest.mark.asyncio
async def test_registration_command_checks_admin_in_its_body(
    control_env, current_image
):
    event = FakeEvent(messages=[current_image], admin=False)
    results = [
        result async for result in control_env.plugin.save_character(event, "Character")
    ]
    assert len(results) == 1
    assert isinstance(results[0], MessageEventResult)
    assert "只有管理员" in results[0].get_plain_text()
    assert control_env.plugin.library.list_names() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("image_count", [0, 2])
async def test_registration_command_requires_exactly_one_image(
    control_env, current_image, image_count
):
    event = FakeEvent(messages=[current_image] * image_count, admin=True)
    results = [
        result async for result in control_env.plugin.save_character(event, "Character")
    ]
    assert "恰好一张" in results[0].get_plain_text()
    assert control_env.plugin.library.list_names() == []


@pytest.mark.asyncio
async def test_registration_command_saves_aliases_and_lists_names(
    control_env, current_image
):
    event = FakeEvent(messages=[current_image], admin=True)
    results = [
        result
        async for result in control_env.plugin.save_character(
            event, "Work Character | Alias | Other Alias"
        )
    ]
    assert "已保存角色" in results[0].get_plain_text()
    saved = control_env.plugin.library.lookup("other alias")
    assert saved["name"] == "Work Character"
    listing = [
        result async for result in control_env.plugin.list_characters(FakeEvent())
    ]
    assert "Work Character" in listing[0].get_plain_text()
    assert saved["image_path"] not in listing[0].get_plain_text()


@pytest.fixture
def search_env(control_env, monkeypatch, current_image):
    """Prepare isolated search results and a real temporary reference image.

    Args:
        control_env: Registered plugin and mocked worker.
        monkeypatch: Fixture restoring download and search functions.
        current_image: A real local PNG fixture.

    Returns:
        The control environment with search, download and verification mocks.
    """
    reference = {
        "image_url": "https://images.example/reference.png",
        "source_website_url": "https://official.example/characters/sherry",
        "caption": "Sherry official character portrait",
    }
    control_env.plugin.config["reference_search"] = "native"
    control_env.provider.search_reference_images.return_value = [reference]
    control_env.provider.verify_reference_image.return_value = True
    control_env.tavily = AsyncMock(return_value=[reference])
    control_env.downloaded = []

    async def download(url, directory):
        """Write a real file in the transient directory to check its lifecycle.

        Args:
            url: Untrusted source URL passed by the plugin.
            directory: Plugin-owned transient directory.

        Returns:
            Path to a real downloaded-image stand-in.
        """
        path = directory / f"candidate-{len(control_env.downloaded)}.png"
        path.write_bytes(Path(await current_image.convert_to_file_path()).read_bytes())
        control_env.downloaded.append(path)
        return path

    control_env.download = AsyncMock(side_effect=download)
    monkeypatch.setattr(
        control_env.module.generation, "download_reference_image", control_env.download
    )
    monkeypatch.setattr(
        control_env.module.generation, "search_tavily_references", control_env.tavily
    )
    return control_env


def test_search_configuration_is_optional_and_lists_supported_backends():
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text("utf-8")
    )
    assert schema["reference_search"]["default"] == "disabled"
    assert schema["reference_search"]["options"] == ["disabled", "native", "tavily"]


@pytest.mark.asyncio
async def test_native_citations_are_extracted_then_visually_checked_before_generation(
    search_env, monkeypatch
):
    env = search_env
    pages = [
        {
            "reference_type": "web_page",
            "source_website_url": "https://official.example/sherry",
            "caption": "Sherry",
        }
    ]
    env.provider.search_reference_images.return_value = pages
    fetch = AsyncMock(
        return_value=(
            b'<html><title>Sherry</title><img src="/art/sherry.png" alt="Sherry portrait"></html>',
            "https://official.example/sherry",
            "text/html",
        )
    )
    monkeypatch.setitem(
        env.module.generation.resolve_reference_pages.__globals__,
        "_fetch_reference_bytes",
        fetch,
    )
    result = await env.plugin.draw_image(
        FakeEvent("Draw Sherry."), "new", ["Sherry"], []
    )
    assert isinstance(result, CallToolResult)
    assert env.download.await_args.args[0] == "https://official.example/art/sherry.png"
    character, request, reference, image_path = (
        env.provider.verify_reference_image.await_args.args
    )
    assert character == "Sherry" and request == "Draw Sherry."
    assert reference["image_url"] != reference["source_website_url"]
    assert reference["source_website_url"] == "https://official.example/sherry"
    assert env.provider.generate_image.await_args.args[1] == [image_path]
    assert "https://official.example/sherry" not in result.content[0].text
    env.provider.search_reference_images.assert_awaited_once()
    env.provider.generate_image.assert_awaited_once()
    env.tavily.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_citations_without_images_fall_back_to_text_only(
    search_env, monkeypatch
):
    env = search_env
    env.provider.search_reference_images.return_value = [
        {
            "reference_type": "web_page",
            "source_website_url": "https://official.example/page",
        }
    ]
    fetch = AsyncMock(
        return_value=(
            b"<html>Login required</html>",
            "https://official.example/page",
            "text/html",
        )
    )
    monkeypatch.setitem(
        env.module.generation.resolve_reference_pages.__globals__,
        "_fetch_reference_bytes",
        fetch,
    )
    event = FakeEvent("Draw Sherry.")
    result = await env.plugin.draw_image(event, "new", ["Sherry"], [])
    assert isinstance(result, CallToolResult)
    env.download.assert_not_awaited()
    env.provider.verify_reference_image.assert_not_awaited()
    env.provider.generate_image.assert_awaited_once()
    assert env.provider.generate_image.await_args.args[1] == []
    assert "已经发起过" in await env.plugin.draw_image(event, "new", ["Sherry"], [])


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["native", "tavily"])
async def test_existing_image_results_skip_page_extraction(
    search_env, monkeypatch, backend
):
    env = search_env
    env.plugin.config["reference_search"] = backend
    extract = AsyncMock(
        side_effect=AssertionError("Image results do not need webpage extraction")
    )
    monkeypatch.setattr(env.module.generation, "resolve_reference_pages", extract)
    assert isinstance(
        await env.plugin.draw_image(FakeEvent("Draw Sherry."), "new", ["Sherry"], []),
        CallToolResult,
    )
    extract.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["native", "tavily"])
async def test_search_passes_only_checked_pixels_and_original_request(
    search_env, backend
):
    env = search_env
    env.plugin.config["reference_search"] = backend
    event = FakeEvent("Draw Sherry from the named work, smiling.")
    await env.plugin.prepare_image_request(
        event, ProviderRequest(contexts=[{"role": "user", "content": "OLD_STYLE"}])
    )
    result = await env.plugin.draw_image(event, "new", ["Sherry"], [])
    assert isinstance(result, CallToolResult)
    if backend == "native":
        env.provider.search_reference_images.assert_awaited_once_with(
            "Sherry", event.text
        )
        env.tavily.assert_not_awaited()
    else:
        env.provider.search_reference_images.assert_not_awaited()
        env.context.get_config.assert_called_once_with(umo=event.unified_msg_origin)
        query, config = env.tavily.await_args.args
        assert "Sherry" in query and event.text in query
        assert config is env.context.get_config.return_value
    env.provider.verify_reference_image.assert_awaited_once()
    prompt, paths = env.provider.generate_image.await_args.args
    assert paths == [str(env.downloaded[0])]
    assert event.text in prompt and "OLD_STYLE" not in prompt
    assert "searched_and_visually_checked_reference" in prompt
    assert "official.example/characters/sherry" not in result.content[0].text
    assert sum(isinstance(part, ImageContent) for part in result.content) == 1
    assert not env.downloaded[0].parent.exists()
    assert env.plugin.library.list_names() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("backend", ["native", "tavily"])
async def test_research_and_drawing_agents_use_distinct_models_and_handoff_checked_evidence(
    search_env, enabled, backend
):
    env = search_env
    research = object.__new__(env.module.ProviderOpenAIResponses)
    research.search_reference_images = AsyncMock(
        return_value=env.provider.search_reference_images.return_value
    )
    research.verify_reference_image = AsyncMock(
        return_value={"appearance": "Blue hair, red ribbon."}
    )
    research.generate_image = AsyncMock()
    env.plugin.config.update(
        {
            "sub_agents_enabled": enabled,
            "research_provider_id": "research",
            "worker_provider_id": "drawing",
            "reference_search": backend,
        }
    )
    env.context.get_provider_by_id.side_effect = {
        "drawing": env.provider,
        "research": research,
    }.get
    event = FakeEvent("Draw Sherry smiling.")
    result = await env.plugin.draw_image(event, "new", ["Sherry"], [])
    assert isinstance(result, CallToolResult)
    researcher = research if enabled else env.provider
    assert researcher.verify_reference_image.await_count == 1
    assert researcher.verify_reference_image.await_args.kwargs == (
        {"collect_details": True} if enabled else {}
    )
    if backend == "native":
        researcher.search_reference_images.assert_awaited_once_with(
            "Sherry", event.text
        )
    else:
        researcher.search_reference_images.assert_not_awaited()
        env.tavily.assert_awaited_once()
    if enabled:
        env.provider.verify_reference_image.assert_not_awaited()
        env.provider.search_reference_images.assert_not_awaited()
    else:
        research.verify_reference_image.assert_not_awaited()
    research.generate_image.assert_not_awaited()
    payload = json.loads(
        env.provider.generate_image.await_args.args[0].rsplit("\n", 1)[1]
    )
    reference = payload["character_references"][0]
    assert reference["character"] == "Sherry" and reference["input_index"] == 1
    assert reference.get("appearance") == (
        "Blue hair, red ribbon." if enabled else None
    )
    assert env.provider.generate_image.await_args.args[1] == [str(env.downloaded[0])]


@pytest.mark.asyncio
async def test_research_agent_rejection_draws_once_without_repeating_search(search_env):
    env = search_env
    env.plugin.config["sub_agents_enabled"] = True
    env.provider.verify_reference_image.return_value = False
    event = FakeEvent("Draw Sherry.")
    result = await env.plugin.draw_image(event, "new", ["Sherry"], [])
    assert isinstance(result, CallToolResult)
    env.provider.generate_image.assert_awaited_once()
    assert env.provider.generate_image.await_args.args[1] == []
    assert "已经发起过" in await env.plugin.draw_image(event, "new", ["Sherry"], [])
    env.provider.search_reference_images.assert_awaited_once()


@pytest.mark.asyncio
async def test_invalid_research_model_skips_search_and_draws_without_references(
    search_env,
):
    env = search_env
    env.plugin.config.update(
        {
            "sub_agents_enabled": True,
            "research_provider_id": "missing",
            "worker_provider_id": "drawing",
        }
    )
    env.context.get_provider_by_id.side_effect = {"drawing": env.provider}.get
    result = await env.plugin.draw_image(
        FakeEvent("Draw Sherry"), "new", ["Sherry"], []
    )
    assert isinstance(result, CallToolResult)
    env.provider.search_reference_images.assert_not_awaited()
    env.provider.generate_image.assert_awaited_once()
    assert env.provider.generate_image.await_args.args[1] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["original", "registered", "current", "edit"])
async def test_optional_research_agent_is_skipped_when_references_are_sufficient(
    search_env, current_image, kind
):
    env = search_env
    env.plugin.config.update(
        {
            "sub_agents_enabled": True,
            "research_provider_id": "missing",
            "worker_provider_id": "drawing",
        }
    )
    env.context.get_provider_by_id.side_effect = {"drawing": env.provider}.get
    event = FakeEvent("Draw Sherry", [current_image])
    names, ids, operation = ["Sherry"], [], "new"
    if kind == "registered":
        await env.plugin.library.save("Sherry", [], current_image)
    elif kind in {"current", "edit"}:
        ids = list(env.module.collect_event_images(event))
        if kind == "edit":
            operation, names = "edit", []
    else:
        event.text, names = "Draw a blue bird", []
    assert isinstance(
        await env.plugin.draw_image(event, operation, names, ids), CallToolResult
    )
    env.context.get_provider_by_id.assert_called_once_with("drawing")
    env.provider.search_reference_images.assert_not_awaited()
    env.provider.verify_reference_image.assert_not_awaited()


@pytest.mark.native_wakeup
@pytest.mark.asyncio
async def test_background_submission_releases_session_and_freezes_inputs(
    background_env,
):
    from astrbot.core.utils.session_lock import session_lock_manager

    env = background_env
    started, release = asyncio.Event(), asyncio.Event()
    captured = []
    generated = env.provider.generate_image.return_value

    async def generate(prompt, images, **options):
        """Block the drawing transport while independent chat can acquire its lock.

        Args:
            prompt: Frozen current image request.
            images: Approved reference paths.
            **options: Image operation and model options.

        Returns:
            The offline generated image after the test releases the worker.
        """
        started.set()
        await release.wait()
        captured.append((prompt, images, options))
        return generated

    env.provider.generate_image.side_effect = generate
    event = env.background_event
    async with session_lock_manager.acquire_lock(event.unified_msg_origin):
        receipt = await asyncio.wait_for(
            env.plugin.draw_image(event, "new", [], []), timeout=1
        )
    job = next(iter(env.plugin._jobs.values()))
    await asyncio.wait_for(started.wait(), timeout=1)
    async with session_lock_manager.acquire_lock(event.unified_msg_origin):
        assert not job["task"].done()
        event.text = "Unrelated later conversation; draw a different character."
        event.set_extra("responses_image_user_text", event.text)
    assert "后台队列" in receipt and env.queue.empty()
    env.context.send_message.assert_not_awaited()
    release.set()
    wakeup = await asyncio.wait_for(env.queue.get(), timeout=2)
    req = env.completion_request(receipt)
    await env.plugin.prepare_image_request(wakeup, req)
    assert req.func_tool.names() == ["responses_draw_image", "send_message_to_user"]
    result = await env.plugin.draw_image(wakeup, "new", [], [])
    assert isinstance(result, CallToolResult)
    assert any(isinstance(part, ImageContent) for part in result.content)
    assert "结果已取回" in await env.plugin.draw_image(wakeup, "new", [], [])
    await asyncio.wait_for(job["task"], timeout=1)
    assert len(captured) == 1 and "Draw a blue bird." in captured[0][0]
    assert "Unrelated later conversation" not in captured[0][0]
    env.context.send_message.assert_not_awaited()
    assert env.plugin._jobs == {}


@pytest.mark.native_wakeup
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["switched", "cleared", "forged_receipt", "missing_tools"]
)
async def test_background_result_cannot_enter_a_changed_or_cleared_conversation(
    background_env, case
):
    env = background_env
    receipt = await env.plugin.draw_image(env.background_event, "new", [], [])
    job = next(iter(env.plugin._jobs.values()))
    wakeup = await asyncio.wait_for(env.queue.get(), timeout=2)
    req = env.completion_request(
        receipt, "different" if case == "switched" else "conversation"
    )
    if case == "cleared":
        req.contexts = []
    elif case == "forged_receipt":
        req.contexts[0]["tool_calls"][0]["function"]["name"] = "unrelated_tool"
    elif case == "missing_tools":
        req.func_tool.remove_tool("send_message_to_user")
    await env.plugin.prepare_image_request(wakeup, req)
    assert wakeup.is_stopped() and job["stop"].is_set()
    assert "失效" in await env.plugin.draw_image(wakeup, "new", [], [])
    await asyncio.wait_for(job["task"], timeout=1)
    env.provider.generate_image.assert_awaited_once()
    env.context.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_skips_wakeup_after_conversation_switch(background_env):
    env = background_env
    env.context.conversation_manager.get_curr_conversation_id.return_value = (
        "new-conversation"
    )
    await env.plugin.draw_image(env.background_event, "new", [], [])
    job = next(iter(env.plugin._jobs.values()))
    await asyncio.wait_for(job["task"], timeout=2)
    assert env.queue.empty() and env.plugin._jobs == {}
    env.context.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_background_submission_is_bounded_and_shutdown_cancels_owned_tasks(
    background_env,
):
    env = background_env
    gate = asyncio.Event()

    async def block_generation(*args, **kwargs):
        """Keep transport pending until task cancellation.

        Args:
            *args: Unused drawing inputs.
            **kwargs: Unused generation options.
        """
        await gate.wait()

    env.provider.generate_image.side_effect = block_generation
    env.plugin.config["max_queue_size"] = 0
    first = await env.plugin.draw_image(env.background_event, "new", [], [])
    assert "后台队列" in first
    assert "已经发起过" in await env.plugin.draw_image(
        env.background_event, "new", [], []
    )
    for index in range(4):
        event = FakeEvent()
        event.platform_meta = env.meta
        event.set_extra("responses_image_conversation_id", "conversation")
        if index:
            event.unified_msg_origin += str(index)
            event.sender_id = f"user-{index}"
        result = await env.plugin.draw_image(event, "new", [], [])
        assert ("已有排队" if index == 0 else "后台队列") in result
    extra = FakeEvent()
    extra.unified_msg_origin += "extra"
    extra.sender_id = "extra-user"
    extra.platform_meta = env.meta
    extra.set_extra("responses_image_conversation_id", "conversation")
    assert "队列已满" in await env.plugin.draw_image(extra, "new", [], [])
    jobs = list(env.plugin._jobs.values())
    await env.plugin.terminate()
    assert env.plugin._jobs == {} and env.queue.empty()
    assert all(job["task"].done() and job["stop"].is_set() for job in jobs)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["unsupported_core", "no_session", "no_proactive"])
async def test_background_preflight_rejects_unsupported_runtime_without_generation(
    background_env, monkeypatch, case
):
    env = background_env
    if case == "unsupported_core":
        monkeypatch.setattr(env.module.background, "wake_main_agent", None)
        monkeypatch.setattr(env.module.background, "run_legacy_completion", None)
    elif case == "no_session":
        env.background_event.set_extra("responses_image_conversation_id", None)
    else:
        env.meta.support_proactive_message = False
    result = await env.plugin.draw_image(env.background_event, "new", [], [])
    assert "关闭后台绘图" in result
    assert env.plugin._jobs == {} and env.queue.empty()
    env.provider.generate_image.assert_not_awaited()


@pytest.mark.native_wakeup
@pytest.mark.asyncio
async def test_background_failure_returns_to_main_agent_without_retry(background_env):
    env = background_env
    env.provider.generate_image.side_effect = RuntimeError("PRIVATE UPSTREAM BODY")
    receipt = await env.plugin.draw_image(env.background_event, "new", [], [])
    job = next(iter(env.plugin._jobs.values()))
    wakeup = await asyncio.wait_for(env.queue.get(), timeout=2)
    await env.plugin.prepare_image_request(wakeup, env.completion_request(receipt))
    result = await env.plugin.draw_image(wakeup, "new", [], [])
    assert "失败" in result and "PRIVATE" not in result
    await asyncio.wait_for(job["task"], timeout=1)
    env.provider.generate_image.assert_awaited_once()


@pytest.mark.native_wakeup
@pytest.mark.asyncio
async def test_background_edit_uses_only_the_submitted_reference_snapshot(
    background_env, current_image, tmp_path
):
    env = background_env
    event = env.background_event
    event.text = "Change this image's background to sunset."
    event.messages = [current_image]
    expected = Path(await current_image.convert_to_file_path()).read_bytes()
    refs = env.module.collect_event_images(event)
    captured = []
    generated = env.provider.generate_image.return_value

    async def edit(prompt, images, **options):
        """Capture uploaded pixels after the source message has changed.

        Args:
            prompt: Isolated edit request.
            images: Approved materialized references.
            **options: Actual edit operation.

        Returns:
            The offline edited image result.
        """
        assert options["operation"] == "edit" and len(images) == 1
        captured.append(Path(images[0]).read_bytes())
        return generated

    env.provider.generate_image.side_effect = edit
    receipt = await env.plugin.draw_image(event, "edit", [], list(refs))
    job = next(iter(env.plugin._jobs.values()))
    current_image.file = str(tmp_path / "generated.png")
    event.messages = []
    wakeup = await asyncio.wait_for(env.queue.get(), timeout=2)
    await env.plugin.prepare_image_request(wakeup, env.completion_request(receipt))
    assert isinstance(
        await env.plugin.draw_image(wakeup, "new", [], []), CallToolResult
    )
    await asyncio.wait_for(job["task"], timeout=1)
    assert captured == [expected]
    env.provider.search_reference_images.assert_not_awaited()


@pytest.mark.native_wakeup
@pytest.mark.asyncio
async def test_plugin_shutdown_invalidates_an_already_queued_completion(background_env):
    env = background_env
    await env.plugin.draw_image(env.background_event, "new", [], [])
    job = next(iter(env.plugin._jobs.values()))
    wakeup = await asyncio.wait_for(env.queue.get(), timeout=2)
    await env.plugin.terminate()
    assert wakeup.get_extra("_background_stop_signal").is_set()
    assert job["result"] is None and env.plugin._jobs == {}
    assert "失效" in await env.plugin.draw_image(wakeup, "new", [], [])
    env.context.send_message.assert_not_awaited()


@pytest.mark.native_wakeup
@pytest.mark.asyncio
async def test_background_timeout_and_unclaimed_result_are_cleaned_up(
    background_env, monkeypatch
):
    env = background_env
    monkeypatch.setattr(env.module.background, "BACKGROUND_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(env.module.background, "RESULT_TIMEOUT_SECONDS", 0.05)
    pending = asyncio.Event()

    async def wait_for_result(plugin, **kwargs):
        """Hold the request until its owner cancels it.

        Args:
            plugin: Owner of the frozen request.
            **kwargs: Ignored validated request arguments.
        """
        await pending.wait()

    monkeypatch.setattr(
        env.module.background,
        "execute_image_request",
        AsyncMock(side_effect=wait_for_result),
    )
    await env.plugin.draw_image(env.background_event, "new", [], [])
    job = next(iter(env.plugin._jobs.values()))
    wakeup = await asyncio.wait_for(env.queue.get(), timeout=1)
    assert "超时" in job["result"]
    await asyncio.wait_for(job["task"], timeout=1)
    assert job["stop"].is_set() and env.plugin._jobs == {}
    assert wakeup.get_extra("_background_stop_signal").is_set()
    env.provider.generate_image.assert_not_awaited()


def test_agent_panel_settings_keep_existing_worker_and_search_configuration():
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text("utf-8")
    )
    assert schema["sub_agents_enabled"]["default"] is False
    assert schema["background_generation"]["default"] is False
    for key in ("research_provider_id", "worker_provider_id"):
        assert schema[key]["_special"] == "select_provider"
        assert schema[key]["default"] == ""
    assert schema["image_model"]["type"] == "string"
    assert schema["image_model"]["default"] == ""
    assert schema["generation_api"]["default"] == "responses"
    assert schema["generation_api"]["options"] == ["responses", "images"]
    assert schema["reference_search"]["options"] == ["disabled", "native", "tavily"]


@pytest.mark.asyncio
@pytest.mark.parametrize("separate_research", [False, True])
async def test_images_api_worker_uses_separate_research_when_needed(
    search_env, monkeypatch, separate_research
):
    env = search_env
    worker = object.__new__(env.module.ProviderOpenAIOfficial)
    env.plugin.config.update(
        {
            "generation_api": "images",
            "worker_provider_id": "drawing",
            "sub_agents_enabled": separate_research,
            "research_provider_id": "research",
        }
    )
    env.context.get_provider_by_id.side_effect = {
        "drawing": worker,
        "research": env.provider,
    }.get
    env.provider.verify_reference_image.return_value = {
        "appearance": "Blue hair, red ribbon."
    }
    binding = Mock(
        return_value=SimpleNamespace(generate_image=env.provider.generate_image)
    )
    monkeypatch.setattr(env.module, "ImagesAPIClient", binding)
    event = FakeEvent("Draw Sherry smiling.")
    result = await env.plugin.draw_image(event, "new", ["Sherry"], [])
    if not separate_research:
        assert isinstance(result, CallToolResult)
        binding.assert_called_once_with(worker)
        env.provider.search_reference_images.assert_not_awaited()
        env.provider.generate_image.assert_awaited_once()
        assert env.provider.generate_image.await_args.args[1] == []
        return
    assert isinstance(result, CallToolResult)
    binding.assert_called_once_with(worker)
    env.provider.search_reference_images.assert_awaited_once()
    assert env.provider.verify_reference_image.await_args.kwargs == {
        "collect_details": True
    }
    prompt, images = env.provider.generate_image.await_args.args
    assert "Blue hair, red ribbon." in prompt
    assert images == [str(env.downloaded[0])]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reference_kind", ["disabled", "registered", "current", "edit", "original"]
)
async def test_search_does_not_replace_existing_reference_flow(
    search_env, current_image, reference_kind
):
    env = search_env
    event = FakeEvent("Draw Sherry.", [current_image])
    character = "Sherry"
    ids = []
    operation = "new"
    if reference_kind == "disabled":
        env.plugin.config["reference_search"] = "disabled"
    elif reference_kind == "registered":
        await env.plugin.library.save("Sherry", [], current_image)
    elif reference_kind in {"current", "edit"}:
        ids = list(env.module.collect_event_images(event))
        if reference_kind == "edit":
            operation, character = "edit", ""
    else:
        character = ""
        event.text = "Draw a blue bird."
    result = await env.plugin.draw_image(
        event, operation, [character] if character else [], ids
    )
    assert isinstance(result, CallToolResult)
    if reference_kind == "disabled":
        assert env.provider.generate_image.await_args.args[1] == []
    env.provider.search_reference_images.assert_not_awaited()
    env.provider.verify_reference_image.assert_not_awaited()
    env.tavily.assert_not_awaited()
    env.download.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_rejects_character_from_history_before_any_request(search_env):
    result = await search_env.plugin.draw_image(
        FakeEvent("Draw her."), "new", ["Sherry"], []
    )
    assert isinstance(result, str)
    search_env.provider.search_reference_images.assert_not_awaited()
    search_env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_checks_next_candidate_but_uses_only_one_accepted_image(
    search_env,
):
    env = search_env
    original = env.provider.search_reference_images.return_value[0]
    env.provider.search_reference_images.return_value = [
        {**original, "image_url": "https://images.example/wrong.png"},
        original,
    ]
    env.provider.verify_reference_image.side_effect = [False, True]
    result = await env.plugin.draw_image(
        FakeEvent("Draw Sherry."), "new", ["Sherry"], []
    )
    assert isinstance(result, CallToolResult)
    assert env.provider.verify_reference_image.await_count == 2
    assert len(env.provider.generate_image.await_args.args[1]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", ["no_results", "download", "unmatched", "verify_error", "search_error"]
)
async def test_search_failure_generates_once_without_references_or_repeat_search(
    search_env, failure
):
    env = search_env
    event = FakeEvent("Draw Sherry.")
    if failure == "no_results":
        env.provider.search_reference_images.return_value = []
    elif failure == "download":
        env.download.side_effect = ValueError("private body")
    elif failure == "unmatched":
        env.provider.verify_reference_image.return_value = False
    elif failure == "verify_error":
        env.provider.verify_reference_image.side_effect = RuntimeError("private body")
    else:
        env.provider.search_reference_images.side_effect = RuntimeError("private body")
    result = await env.plugin.draw_image(event, "new", ["Sherry"], [])
    assert isinstance(result, CallToolResult)
    assert "private body" not in result.content[0].text
    second = await env.plugin.draw_image(event, "new", ["Sherry"], [])
    assert "已经发起过" in second
    env.provider.search_reference_images.assert_awaited_once()
    env.provider.generate_image.assert_awaited_once()
    assert env.provider.generate_image.await_args.args[1] == []
    assert env.plugin.quota.snapshot(event.unified_msg_origin)["used_hour"] == 1
    assert not list(env.plugin.library.data_dir.glob("search-*"))


@pytest.mark.asyncio
async def test_cancelled_verification_cleans_downloads_and_keeps_attempt_marker(
    search_env,
):
    env = search_env
    event = FakeEvent("Draw Sherry.")
    started = asyncio.Event()
    pending = asyncio.Event()

    async def verify(*args):
        """Pause verification until cancellation.

        Args:
            *args: The original request and downloaded reference.
        """
        started.set()
        await pending.wait()

    env.provider.verify_reference_image.side_effect = verify
    task = asyncio.create_task(env.plugin.draw_image(event, "new", ["Sherry"], []))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert event.get_extra("responses_image_attempted") is True
    assert not event.get_extra("responses_image_lock").locked()
    assert not list(env.plugin.library.data_dir.glob("search-*"))
    env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_pipeline_is_deduplicated_under_concurrent_tool_calls(search_env):
    env = search_env
    event = FakeEvent("Draw Sherry.")
    results = await asyncio.gather(
        env.plugin.draw_image(event, "new", ["Sherry"], []),
        env.plugin.draw_image(event, "new", ["Sherry"], []),
    )
    assert sum(isinstance(result, CallToolResult) for result in results) == 1
    env.provider.search_reference_images.assert_awaited_once()
    env.provider.generate_image.assert_awaited_once()


@pytest_asyncio.fixture
async def registered_pair(control_env, current_image, tmp_path):
    """Register two distinct character images for target-to-image mapping tests.

    Args:
        control_env: Plugin and mocked worker fixture.
        current_image: First character's reference image.
        tmp_path: Isolated location for the second image.

    Returns:
        The control environment with two named character profiles.
    """
    path = tmp_path / "second.png"
    PillowImage.new("RGB", (3, 2), "blue").save(path)
    control_env.pair = [
        await control_env.plugin.library.save("蕾米", ["Remilia"], current_image),
        await control_env.plugin.library.save(
            "芙兰", ["Flandre"], Comp.Image.fromFileSystem(path)
        ),
    ]
    return control_env


@pytest.mark.asyncio
async def test_context_pronoun_resolves_two_characters_without_forwarding_history(
    registered_pair,
):
    env = registered_pair
    event = FakeEvent("画她们一起喝茶。")
    history = [
        {"role": "user", "content": "想画蕾米和芙兰。OLD_UNRELATED_APPEARANCE"},
        {"role": "assistant", "content": "可以让两人同框。"},
    ]
    await env.plugin.prepare_image_request(event, ProviderRequest(contexts=history))
    snapshots = event.get_extra("responses_image_contexts")
    selected = [key for key, value in snapshots.items() if value["role"] == "user"]
    result = await env.plugin.draw_image(event, "new", ["蕾米", "芙兰"], [], selected)
    assert isinstance(result, CallToolResult)
    prompt, paths = env.provider.generate_image.await_args.args
    data = json.loads(prompt.rsplit("\n", 1)[1])
    assert data["user_request"] == event.text
    assert data["characters"] == ["蕾米", "芙兰"]
    assert paths == [profile["image_path"] for profile in env.pair]
    assert [
        (item["character"], item["input_index"])
        for item in data["character_references"]
    ] == [("蕾米", 1), ("芙兰", 2)]
    assert "OLD_UNRELATED_APPEARANCE" not in prompt
    assert "unrelated-old-image" not in prompt
    assert all(key not in prompt for key in selected)
    env.provider.search_reference_images.assert_not_awaited()


@pytest.mark.asyncio
async def test_assistant_text_can_supply_target_identity_but_not_old_style(
    registered_pair,
):
    env = registered_pair
    event = FakeEvent("那画她吧。")
    request = ProviderRequest(
        contexts=[{"role": "assistant", "content": "这个角色是 Remilia。OLD_STYLE"}]
    )
    await env.plugin.prepare_image_request(event, request)
    ids = list(event.get_extra("responses_image_contexts"))
    request.contexts[0]["content"] = "Changed after the request snapshot"
    result = await env.plugin.draw_image(event, "new", ["蕾米"], [], ids)
    assert isinstance(result, CallToolResult)
    prompt, paths = env.provider.generate_image.await_args.args
    assert paths == [env.pair[0]["image_path"]]
    assert "OLD_STYLE" not in prompt and "Changed after" not in prompt


@pytest.mark.asyncio
async def test_current_explicit_character_cannot_be_replaced_by_selected_history(
    registered_pair,
):
    env = registered_pair
    event = FakeEvent("这次画芙兰。")
    await env.plugin.prepare_image_request(
        event, ProviderRequest(contexts=[{"role": "user", "content": "之前画蕾米。"}])
    )
    ids = list(event.get_extra("responses_image_contexts"))
    result = await env.plugin.draw_image(event, "new", ["蕾米"], [], ids)
    assert isinstance(result, str)
    env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_multiple_explicit_characters_are_selected_without_context(
    registered_pair,
):
    env = registered_pair
    result = await env.plugin.draw_image(
        FakeEvent("画蕾米和芙兰。"), "new", ["蕾米", "芙兰"], []
    )
    assert isinstance(result, CallToolResult)
    data = json.loads(env.provider.generate_image.await_args.args[0].rsplit("\n", 1)[1])
    assert set(data["characters"]) == {"蕾米", "芙兰"}
    assert len(data["character_references"]) == 2


@pytest.mark.asyncio
async def test_empty_targets_with_selected_context_require_tool_argument_resolution(
    registered_pair,
):
    env = registered_pair
    event = FakeEvent("画她们。")
    await env.plugin.prepare_image_request(
        event, ProviderRequest(contexts=[{"role": "user", "content": "蕾米和芙兰"}])
    )
    result = await env.plugin.draw_image(
        event, "new", [], [], list(event.get_extra("responses_image_contexts"))
    )
    assert isinstance(result, str)
    assert "character_names" in result
    env.provider.generate_image.assert_not_awaited()
    env.provider.search_reference_images.assert_not_awaited()


@pytest.mark.asyncio
async def test_multiple_current_names_need_model_selection_when_targets_are_empty(
    registered_pair,
):
    env = registered_pair
    result = await env.plugin.draw_image(
        FakeEvent("只画蕾米，不要芙兰。"), "new", [], []
    )
    assert isinstance(result, str)
    assert "character_names" in result
    env.provider.generate_image.assert_not_awaited()
    env.provider.search_reference_images.assert_not_awaited()


@pytest.mark.asyncio
async def test_context_aliases_of_one_character_share_one_reference(registered_pair):
    env = registered_pair
    event = FakeEvent("画她吧。")
    await env.plugin.prepare_image_request(
        event,
        ProviderRequest(contexts=[{"role": "user", "content": "Remilia 就是蕾米。"}]),
    )
    result = await env.plugin.draw_image(
        event,
        "new",
        ["Remilia", "蕾米"],
        [],
        list(event.get_extra("responses_image_contexts")),
    )
    assert isinstance(result, CallToolResult)
    prompt, paths = env.provider.generate_image.await_args.args
    assert paths == [env.pair[0]["image_path"]]
    assert json.loads(prompt.rsplit("\n", 1)[1])["characters"] == ["蕾米"]


@pytest.mark.asyncio
async def test_context_snapshot_keeps_only_conversation_text(control_env):
    event = FakeEvent("画她们。")
    history = [
        {"role": "system", "content": "SYSTEM_ONLY_CHARACTER"},
        {"role": "tool", "content": "TOOL_ONLY_CHARACTER"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "tool"}],
            "content": "TOOL_PLAN_CHARACTER",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "User character"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,OLD_PIXELS"},
                },
                {"type": "reasoning", "text": "PRIVATE_REASONING_CHARACTER"},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Assistant character"}],
        },
    ]
    request = ProviderRequest(contexts=copy.deepcopy(history))
    await control_env.plugin.prepare_image_request(event, request)
    snapshots = event.get_extra("responses_image_contexts")
    assert len(snapshots) == 2
    assert {item["text"] for item in snapshots.values()} == {
        "User character",
        "Assistant character",
    }
    assert request.contexts == history
    serialized = json.dumps(snapshots)
    for forbidden in (
        "SYSTEM_ONLY",
        "TOOL_ONLY",
        "TOOL_PLAN",
        "OLD_PIXELS",
        "PRIVATE_REASONING",
    ):
        assert forbidden not in serialized
        assert forbidden not in request.system_prompt
    for key in snapshots:
        assert key in request.system_prompt


@pytest.mark.asyncio
async def test_context_snapshot_bounds_recent_text_and_scan_window(control_env):
    event = FakeEvent()
    request = ProviderRequest(
        contexts=[
            {"role": "user", "content": f"MESSAGE_{index}:" + "x" * 3000}
            for index in range(12)
        ]
    )
    await control_env.plugin.prepare_image_request(event, request)
    snapshots = event.get_extra("responses_image_contexts")
    assert 1 <= len(snapshots) <= 8
    assert sum(len(item["text"]) for item in snapshots.values()) <= 8000
    assert all(len(item["text"]) <= 2000 for item in snapshots.values())
    old_event = FakeEvent()
    old_request = ProviderRequest(
        contexts=[
            {"role": "user", "content": "OUTSIDE_SCAN_WINDOW"},
            *[{"role": "tool", "content": "tool result"} for _ in range(40)],
        ]
    )
    await control_env.plugin.prepare_image_request(old_event, old_request)
    assert not old_event.get_extra("responses_image_contexts")


@pytest.mark.asyncio
async def test_context_ids_are_event_local_and_cannot_be_replayed(registered_pair):
    env = registered_pair
    history = [{"role": "user", "content": "蕾米"}]
    previous, current = FakeEvent("画她。"), FakeEvent("再画她。")
    await env.plugin.prepare_image_request(previous, ProviderRequest(contexts=history))
    await env.plugin.prepare_image_request(current, ProviderRequest(contexts=history))
    stale_ids = list(previous.get_extra("responses_image_contexts"))
    assert set(stale_ids).isdisjoint(current.get_extra("responses_image_contexts"))
    result = await env.plugin.draw_image(current, "new", ["蕾米"], [], stale_ids)
    assert isinstance(result, str)
    env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_selected_context_can_supply_unknown_target(search_env):
    env = search_env
    event = FakeEvent("Draw her.")
    await env.plugin.prepare_image_request(
        event,
        ProviderRequest(
            contexts=[
                {"role": "user", "content": "Sherry"},
                {"role": "assistant", "content": "Another character"},
            ]
        ),
    )
    unrelated_ids = [
        key
        for key, value in event.get_extra("responses_image_contexts").items()
        if value["role"] == "assistant"
    ]
    result = await env.plugin.draw_image(event, "new", ["Sherry"], [], unrelated_ids)
    assert isinstance(result, str)
    env.provider.search_reference_images.assert_not_awaited()
    env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_original_subject_does_not_implicitly_adopt_historical_characters(
    registered_pair,
):
    env = registered_pair
    event = FakeEvent("画一只小鸟。")
    await env.plugin.prepare_image_request(
        event, ProviderRequest(contexts=[{"role": "user", "content": "蕾米和芙兰"}])
    )
    result = await env.plugin.draw_image(event, "new", [], [])
    assert isinstance(result, CallToolResult)
    prompt, paths = env.provider.generate_image.await_args.args
    assert paths == []
    assert "蕾米" not in prompt and "芙兰" not in prompt


@pytest.mark.asyncio
async def test_context_character_matching_uses_ascii_word_boundaries(search_env):
    env = search_env
    event = FakeEvent("Draw her.")
    await env.plugin.prepare_image_request(
        event, ProviderRequest(contexts=[{"role": "user", "content": "Anna"}])
    )
    result = await env.plugin.draw_image(
        event, "new", ["Ann"], [], list(event.get_extra("responses_image_contexts"))
    )
    assert isinstance(result, str)
    env.provider.search_reference_images.assert_not_awaited()


@pytest.mark.asyncio
async def test_context_pair_searches_and_binds_each_target_separately(search_env):
    env = search_env
    event = FakeEvent("画她们一起喝茶。")
    await env.plugin.prepare_image_request(
        event,
        ProviderRequest(
            contexts=[
                {
                    "role": "user",
                    "content": "蕾米和芙兰。OLD_APPEARANCE_MUST_STAY_IN_CHAT",
                }
            ]
        ),
    )
    first = env.provider.search_reference_images.return_value[0]
    second = {
        **first,
        "image_url": "https://images.example/flandre.png",
        "source_website_url": "https://official.example/flandre",
    }
    env.provider.search_reference_images.side_effect = [[first], [second]]
    result = await env.plugin.draw_image(
        event,
        "new",
        ["蕾米", "芙兰"],
        [],
        list(event.get_extra("responses_image_contexts")),
    )
    assert isinstance(result, CallToolResult)
    assert [
        call.args for call in env.provider.search_reference_images.await_args_list
    ] == [("蕾米", event.text), ("芙兰", event.text)]
    assert [
        call.args[0] for call in env.provider.verify_reference_image.await_args_list
    ] == ["蕾米", "芙兰"]
    prompt, paths = env.provider.generate_image.await_args.args
    data = json.loads(prompt.rsplit("\n", 1)[1])
    assert paths == [str(path) for path in env.downloaded]
    assert len(set(paths)) == 2
    assert [
        (item["character"], item["input_index"])
        for item in data["character_references"]
    ] == [("蕾米", 1), ("芙兰", 2)]
    assert "OLD_APPEARANCE_MUST_STAY_IN_CHAT" not in prompt
    assert "official.example/flandre" not in result.content[0].text
    assert not list(env.plugin.library.data_dir.glob("search-*"))


@pytest.mark.asyncio
async def test_missing_second_character_reference_clears_all_automatic_references(
    search_env,
):
    env = search_env
    event = FakeEvent("画蕾米和芙兰。")
    first = env.provider.search_reference_images.return_value
    env.provider.search_reference_images.side_effect = [first, []]
    result = await env.plugin.draw_image(event, "new", ["蕾米", "芙兰"], [])
    assert isinstance(result, CallToolResult)
    assert env.provider.search_reference_images.await_count == 2
    env.provider.generate_image.assert_awaited_once()
    prompt, paths = env.provider.generate_image.await_args.args
    assert paths == []
    assert json.loads(prompt.rsplit("\n", 1)[1])["character_references"] == []
    assert not list(env.plugin.library.data_dir.glob("search-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "names", ["Sherry", [None], [""], ["x" * 201], ["A", "B", "C", "D", "E"]]
)
async def test_character_list_rejects_invalid_or_unbounded_arguments(
    control_env, names
):
    result = await control_env.plugin.draw_image(
        FakeEvent("Draw Sherry."), "new", names, []
    )
    assert isinstance(result, str)
    control_env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("context_ids", ["ctx_forged", [None], ["ctx_forged"]])
async def test_invalid_context_selection_never_reaches_worker(control_env, context_ids):
    result = await control_env.plugin.draw_image(
        FakeEvent(), "new", [], [], context_ids
    )
    assert isinstance(result, str)
    control_env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "expected_hint"),
    [
        ("missing_results", "web_search_call.results"),
        ("search_not_called", "实际未执行原生搜索"),
        ("no_image_results", "没有 image_result"),
        ("no_usable_reference", "图片地址或来源网页"),
        ("incomplete_response", "没有正常完成"),
        ("private-body-with-token", "原生搜索响应不满足图片参考要求"),
    ],
)
async def test_native_search_errors_fall_back_without_exposing_raw_errors(
    search_env, reason, expected_hint
):
    env = search_env
    env.provider.search_reference_images.side_effect = (
        env.module.generation.ReferenceSearchError(reason)
    )
    event = FakeEvent("Draw Sherry.")
    result = await env.plugin.draw_image(event, "new", ["Sherry"], [])
    assert isinstance(result, CallToolResult)
    assert expected_hint not in result.content[0].text
    assert "private-body-with-token" not in result.content[0].text
    env.provider.generate_image.assert_awaited_once()
    assert env.provider.generate_image.await_args.args[1] == []
    env.provider.verify_reference_image.assert_not_awaited()
    second = await env.plugin.draw_image(event, "new", ["Sherry"], [])
    assert "已经发起过" in second
    env.provider.search_reference_images.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_kind", ["native", "generic"])
async def test_search_diagnostics_are_never_returned_to_chat(search_env, error_kind):
    error = (
        RuntimeError("PRIVATE_BODY")
        if error_kind == "generic"
        else search_env.module.generation.ReferenceSearchError("missing_results")
    )
    error.diagnostic = '{"raw_sdk_match":true,"diagnostic_marker":"PRIVATE_DATA"}'
    search_env.provider.search_reference_images.side_effect = error
    result = await search_env.plugin.draw_image(
        FakeEvent("Draw Sherry."), "new", ["Sherry"], []
    )
    assert isinstance(result, CallToolResult)
    assert "raw_sdk_match" not in result.content[0].text
    assert "diagnostic_marker" not in result.content[0].text
    assert "PRIVATE_" not in result.content[0].text
    search_env.provider.generate_image.assert_awaited_once()
