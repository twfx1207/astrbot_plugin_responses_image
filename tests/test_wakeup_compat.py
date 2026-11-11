"""Exercise native selection and released-core proactive image completion."""

import asyncio
import base64
import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import pytest_asyncio
from astrbot.core.message import components as Comp
from astrbot.core.provider.entities import LLMResponse, ProviderRequest
from test_control import FakeEvent
from test_control import background_env as background_env
from test_control import control_env as control_env


def test_backend_selection_matches_installed_core(control_env):
    module = control_env.module
    if importlib.util.find_spec("astrbot.core.astr_agent_wakeup") is not None:
        assert callable(module.background.wake_main_agent)
        assert module.background.native_wakeup_import_error is None
        assert module.background.run_legacy_completion is None
    else:
        assert module.background.wake_main_agent is None
        assert (
            module.background.native_wakeup_import_error[1]
            == "astrbot.core.astr_agent_wakeup"
        )
        assert callable(module.background.run_legacy_completion), (
            module.background.legacy_wakeup_import_error
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_native_wakeup_is_preferred_without_runtime_fallback(
    background_env, monkeypatch, failed
):
    env = background_env
    if env.module.background.wake_main_agent is None:
        pytest.skip("Requires a core containing the native wakeup API")
    legacy = AsyncMock()
    monkeypatch.setattr(env.module.background, "run_legacy_completion", legacy)
    if failed:
        monkeypatch.setattr(
            env.module.background,
            "wake_main_agent",
            AsyncMock(side_effect=RuntimeError("private body")),
        )
    receipt = await env.plugin.draw_image(env.background_event, "new", [], [])
    job = next(iter(env.plugin._jobs.values()))
    if not failed:
        event = await asyncio.wait_for(env.queue.get(), 2)
        await env.plugin.prepare_image_request(event, env.completion_request(receipt))
        await env.plugin.draw_image(event, "new", [], [])
    await asyncio.wait_for(job["task"], 2)
    legacy.assert_not_awaited()
    env.provider.generate_image.assert_awaited_once()
    assert not env.plugin._jobs


@pytest_asyncio.fixture
async def legacy_env(background_env, monkeypatch, tmp_path):
    """Run the released builder, tool runner, sender and history saver offline.

    Args:
        background_env: Isolated plugin with a simulated image transport.
        monkeypatch: Restores hooks and per-session permission lookup.
        tmp_path: Temporary image cache location.

    Returns:
        Environment with real core execution and mocked model/network boundaries.
    """
    env = background_env
    if env.module.background.wake_main_agent is not None:
        pytest.skip("Run with RESPONSES_IMAGE_TEST_CORE pointing to v4.29.4")
    assert env.module.background.run_legacy_completion is not None, (
        env.module.background.legacy_wakeup_import_error
    )
    import astrbot.core.astr_agent_hooks as hooks
    import astrbot.core.astr_main_agent as builder
    from astrbot.core.agent.tool_image_cache import tool_image_cache
    from astrbot.core.star.context import Context
    from astrbot.core.star.star_handler import EventType
    from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

    context = object.__new__(Context)
    context.__dict__.update(vars(env.context))
    env.context = env.plugin.context = context
    env.legacy = importlib.import_module(env.module.__package__ + ".legacy_completion")
    env.completion_errors = []
    complete = env.module.background.run_legacy_completion

    async def observed_completion(*args):
        """Keep raised errors visible after the background worker handles them.

        Args:
            *args: Completion inputs for the actual legacy implementation.
        """
        try:
            await complete(*args)
        except Exception as exc:
            env.completion_errors.append(exc)
            raise

    monkeypatch.setattr(
        env.module.background, "run_legacy_completion", observed_completion
    )
    env.cfg = {
        "provider_settings": {"enable": True, "streaming_response": True},
        "agent_runner": {
            "runner_type": "local",
            "config": {
                "persona": {"persona_id": "chosen-persona"},
                "model": {"request_max_retries": 0},
                "misc": {"max_steps": 4},
            },
        },
    }
    env.context.get_config.return_value = env.cfg
    env.conversation = SimpleNamespace(
        cid="conversation",
        history="[]",
        persona_id="chosen-persona",
        token_usage=0,
    )
    env.context.conversation_manager.get_conversation = AsyncMock(
        return_value=env.conversation
    )
    env.context.conversation_manager.update_conversation = AsyncMock()
    env.context.persona_manager = SimpleNamespace(
        resolve_selected_persona=AsyncMock(
            return_value=(
                "chosen-persona",
                {"prompt": "KNOWN_PERSONA", "tools": None},
                False,
                False,
            ),
        )
    )
    env.context.subagent_orchestrator = None
    env.context.get_llm_tool_manager = Mock(return_value=env.tools)
    env.tools.get_func("responses_draw_image").handler = env.plugin.draw_image
    env.allowed = AsyncMock(return_value=True)
    monkeypatch.setattr(
        env.legacy.SessionServiceManager, "should_process_llm_request", env.allowed
    )
    monkeypatch.setattr(
        builder, "retrieve_knowledge_base", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        tool_image_cache,
        "_cache_dir",
        str(Path(get_astrbot_temp_path()) / "legacy-tests" / tmp_path.name),
    )
    env.build = AsyncMock(wraps=env.legacy.build_main_agent)
    monkeypatch.setattr(env.legacy, "build_main_agent", env.build)
    env.requests = []
    env.events = []

    async def hook(event, event_type, *args):
        """Dispatch real main-agent callbacks to the isolated plugin.

        Args:
            event: Fresh completion event.
            event_type: AstrBot hook type.
            *args: Provider request or tool execution arguments.
        """
        if event_type == EventType.OnLLMRequestEvent:
            env.events.append(event)
            env.requests.append(args[0])
            await env.plugin.prepare_image_request(event, *args)
        elif event_type == EventType.OnUsingLLMToolEvent:
            await env.plugin.restrict_image_delivery(event, *args)
        return event.is_stopped()

    monkeypatch.setattr(env.legacy, "call_event_hook", hook)
    monkeypatch.setattr(hooks, "call_event_hook", hook)
    env.main = object.__new__(env.module.ProviderOpenAIResponses)
    env.main.provider_config = {
        "id": "main",
        "modalities": ["text", "image", "tool_use"],
        "max_context_tokens": 128000,
    }
    env.main.model_name = "offline-main"
    env.main.client = SimpleNamespace(base_url=httpx.URL("https://example.invalid/v1"))
    env.context.get_using_provider_async = AsyncMock(return_value=env.main)
    env.main_calls = []
    env.viewed = []
    env.include_caption = True

    async def respond(**kwargs):
        """Review pixels before requesting delivery through the actual sender.

        Args:
            **kwargs: Actual requests assembled by the core runner.

        Returns:
            Controlled model responses, without invoking an external API.
        """
        env.main_calls.append(kwargs)
        if len(env.main_calls) == 1:
            assert kwargs["func_tool"].names() == [
                "responses_draw_image",
                "send_message_to_user",
            ]
            assert any(
                item.get("role") == "system"
                and "KNOWN_PERSONA" in item.get("content", "")
                for item in kwargs["contexts"]
            )
            return LLMResponse(
                role="assistant",
                tools_call_name=["responses_draw_image"],
                tools_call_args=[
                    {"operation": "new", "character_names": [], "reference_ids": []}
                ],
                tools_call_ids=["fetch_image"],
            )
        if len(env.main_calls) == 2:
            env.context.send_message.assert_not_awaited()
            images = [
                part["image_url"]
                for message in kwargs["contexts"]
                for part in (
                    message.get("content")
                    if isinstance(message.get("content"), list)
                    else []
                )
                if part.get("type") == "image_url"
            ]
            assert len(images) == 1
            pixels = base64.b64decode(images[0]["url"].split(",", 1)[1])
            assert pixels == (tmp_path / "generated.png").read_bytes()
            env.viewed.append(images[0]["id"])
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
                            {
                                "type": "plain",
                                "text": "画好了，使用了蓝色背景。"
                                if env.include_caption
                                else "",
                            },
                            {"type": "mention_user", "mention_user_id": "wrong-person"},
                            {"type": "image", "path": env.viewed[0]},
                        ]
                    }
                ],
                tools_call_ids=["deliver_image"],
            )
        pytest.fail("Successful image delivery must not call the model again")

    env.main.text_chat = AsyncMock(side_effect=respond)
    return env


@pytest.mark.asyncio
@pytest.mark.parametrize("group", [False, True])
@pytest.mark.parametrize("caption", [False, True])
async def test_legacy_completion_reviews_sends_and_persists_editable_pixels(
    legacy_env, group, caption
):
    from astrbot.core.utils.session_lock import session_lock_manager

    env = legacy_env
    env.include_caption = caption
    if group:
        env.background_event.unified_msg_origin = "qq:GroupMessage:group!isolated-user"
    umo = env.background_event.unified_msg_origin
    snapshot = copy.deepcopy(env.cfg)
    async with session_lock_manager.acquire_lock(umo):
        receipt = await env.plugin.draw_image(env.background_event, "new", [], [])
        env.background_event.sender_id = "later-speaker"
        job = next(iter(env.plugin._jobs.values()))
        env.conversation.history = json.dumps(env.completion_request(receipt).contexts)
    await asyncio.wait_for(job["task"], 5)
    if env.completion_errors:
        raise env.completion_errors[0]
    assert env.queue.empty() and not env.plugin._jobs
    assert len(env.main_calls) == 2
    assert job["claimed"].is_set() and not job["stop"].is_set()
    event = env.events[0]
    assert event.unified_msg_origin == umo
    assert event.get_platform_name() == "qq_official" and not event.is_admin()
    assert event.message_obj.message_id
    assert event.get_extra("enable_streaming") is False
    env.context.persona_manager.resolve_selected_persona.assert_awaited_once()
    assert (
        env.context.persona_manager.resolve_selected_persona.await_args.kwargs[
            "platform_name"
        ]
        == "qq_official"
    )
    assert env.requests[0].system_prompt.count("[Background image completion]") == 1
    assert env.cfg == snapshot
    assert env.context.send_message.await_count == (2 if group else 1)
    target, chain = env.context.send_message.await_args_list[0].args
    assert str(target) == umo
    assert len(chain.chain) == 1 + (caption and not group)
    assert isinstance(chain.chain[-1], Comp.Image)
    assert not any(isinstance(part, Comp.At) for part in chain.chain)
    assert chain.get_plain_text() == (
        "画好了，使用了蓝色背景。" if caption and not group else ""
    )
    if group:
        target, notice = env.context.send_message.await_args_list[1].args
        # Unknown connector capabilities use plain text, never speculative tags.
        assert str(target) == umo and notice.use_markdown_ is False
        assert not any(isinstance(part, Comp.At) for part in notice.chain)
        assert notice.get_plain_text().strip() == (
            "画好了，使用了蓝色背景。" if caption else "图片已生成。"
        )
        assert not any(isinstance(part, Comp.Image) for part in notice.chain)
    saved = env.context.conversation_manager.update_conversation.await_args.kwargs[
        "history"
    ]
    Path(env.viewed[0]).unlink()
    edit_event = FakeEvent("修改刚才那张图，让背景更暗。")
    await env.plugin.prepare_image_request(edit_event, ProviderRequest(contexts=saved))
    assert len(edit_event.get_extra("responses_image_generated_references")) == 1
    env.provider.generate_image.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "cleared",
        "forged_receipt",
        "disabled_tool",
        "disabled_plugin",
        "disabled_llm",
        "session_disabled",
        "switched",
    ],
)
async def test_legacy_completion_revalidates_after_acquiring_session_lock(
    legacy_env, case
):
    from astrbot.core.utils.session_lock import session_lock_manager

    env = legacy_env
    async with session_lock_manager.acquire_lock(
        env.background_event.unified_msg_origin
    ):
        receipt = await env.plugin.draw_image(env.background_event, "new", [], [])
        job = next(iter(env.plugin._jobs.values()))
        history = env.completion_request(receipt).contexts
        if case == "cleared":
            history = []
        elif case == "forged_receipt":
            history[0]["tool_calls"][0]["function"]["name"] = "unrelated_tool"
        elif case == "disabled_tool":
            env.tools.get_func("responses_draw_image").active = False
        elif case == "disabled_plugin":
            env.cfg["plugin_set"] = []
        elif case == "disabled_llm":
            env.cfg["provider_settings"]["enable"] = False
        elif case == "session_disabled":
            env.allowed.return_value = False
        elif case == "switched":
            env.context.conversation_manager.get_curr_conversation_id.return_value = (
                "new-conversation"
            )
        env.conversation.history = json.dumps(history)
    await asyncio.wait_for(job["task"], 5)
    env.main.text_chat.assert_not_awaited()
    env.context.send_message.assert_not_awaited()
    env.context.conversation_manager.update_conversation.assert_not_awaited()
    assert not env.plugin._jobs and not env.plugin.quota.reservations
    assert not env.completion_errors


@pytest.mark.asyncio
async def test_legacy_rejects_external_agent_before_spending_quota(legacy_env):
    env = legacy_env
    env.cfg["agent_runner"]["runner_type"] = "coze"
    reply = await env.plugin.draw_image(env.background_event, "new", [], [])
    assert "本地 Agent" in reply
    assert not env.plugin._jobs
    env.provider.generate_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_generation_failure_returns_one_safe_explanation(legacy_env):
    from astrbot.core.utils.session_lock import session_lock_manager

    env = legacy_env
    env.provider.generate_image.side_effect = RuntimeError("PRIVATE_UPSTREAM_BODY")
    env.main.text_chat.side_effect = [
        LLMResponse(
            role="assistant",
            tools_call_name=["responses_draw_image"],
            tools_call_args=[
                {"operation": "new", "character_names": [], "reference_ids": []}
            ],
            tools_call_ids=["fetch_error"],
        ),
        LLMResponse(role="assistant", completion_text="这次绘图失败，没有自动重试。"),
    ]
    async with session_lock_manager.acquire_lock(
        env.background_event.unified_msg_origin
    ):
        receipt = await env.plugin.draw_image(env.background_event, "new", [], [])
        job = next(iter(env.plugin._jobs.values()))
        env.conversation.history = json.dumps(env.completion_request(receipt).contexts)
    await asyncio.wait_for(job["task"], 5)
    assert not env.completion_errors
    env.context.send_message.assert_awaited_once()
    chain = env.context.send_message.await_args.args[1]
    assert chain.get_plain_text() == "这次绘图失败，没有自动重试。"
    assert "PRIVATE" not in str(env.main.text_chat.await_args)
    env.provider.generate_image.assert_awaited_once()
    assert (
        env.plugin.quota.snapshot(env.background_event.unified_msg_origin)["used_hour"]
        == 1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["unload", "reset"])
async def test_legacy_active_model_call_is_cancelled_without_sending(legacy_env, stop):
    from astrbot.core.utils.active_event_registry import active_event_registry
    from astrbot.core.utils.session_lock import session_lock_manager

    env = legacy_env
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def block_model(**kwargs):
        """Keep the real runner in a cancellable model transport call.

        Args:
            **kwargs: Main model request.
        """
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    env.main.text_chat.side_effect = block_model
    umo = env.background_event.unified_msg_origin
    async with session_lock_manager.acquire_lock(umo):
        receipt = await env.plugin.draw_image(env.background_event, "new", [], [])
        job = next(iter(env.plugin._jobs.values()))
        env.conversation.history = json.dumps(env.completion_request(receipt).contexts)
    await asyncio.wait_for(entered.wait(), 3)
    if stop == "unload":
        await env.plugin.terminate()
    else:
        active_event_registry.stop_all(umo)
    await asyncio.wait_for(asyncio.gather(job["task"], return_exceptions=True), 2)
    assert cancelled.is_set() and not env.plugin._jobs
    env.context.send_message.assert_not_awaited()
    env.context.conversation_manager.update_conversation.assert_not_awaited()
    assert not env.completion_errors


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["unload", "timeout", "reset"])
async def test_legacy_waiting_completion_stops_without_delayed_delivery(
    legacy_env, monkeypatch, stop
):
    from astrbot.core.utils.active_event_registry import active_event_registry
    from astrbot.core.utils.session_lock import session_lock_manager

    env = legacy_env
    if stop == "timeout":
        monkeypatch.setattr(env.module.background, "RESULT_TIMEOUT_SECONDS", 0.05)
    entered = asyncio.Event()
    original = env.module.background.run_legacy_completion

    async def complete(*args):
        """Signal that generation finished before waiting for the session lock.

        Args:
            *args: Validated completion inputs.
        """
        entered.set()
        await original(*args)

    monkeypatch.setattr(env.module.background, "run_legacy_completion", complete)
    umo = env.background_event.unified_msg_origin
    async with session_lock_manager.acquire_lock(umo):
        receipt = await env.plugin.draw_image(env.background_event, "new", [], [])
        job = next(iter(env.plugin._jobs.values()))
        env.conversation.history = json.dumps(env.completion_request(receipt).contexts)
        await asyncio.wait_for(entered.wait(), 2)
        if stop == "unload":
            await env.plugin.terminate()
        elif stop == "timeout":
            await asyncio.wait_for(job["task"], 2)
        else:
            active_event_registry.stop_all(umo)
    await asyncio.wait_for(asyncio.gather(job["task"], return_exceptions=True), 2)
    env.main.text_chat.assert_not_awaited()
    env.context.send_message.assert_not_awaited()
    assert not env.plugin._jobs
