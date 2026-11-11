"""Verify that plugin activation never owns or registers model providers."""

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from astrbot.core.message.components import Image
from astrbot.core.provider.provider import Provider
from PIL import Image as PillowImage


@pytest.fixture
def lifecycle_env(monkeypatch, tmp_path):
    """Load the plugin with isolated existing provider registrations.

    Args:
        monkeypatch: Fixture restoring modules and shared registries.
        tmp_path: Character library directory.

    Yields:
        The plugin module, internal clients, registry and existing entries.
    """
    package_name = "responses_image_lifecycle_under_test"
    package = ModuleType(package_name)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    monkeypatch.setitem(sys.modules, package_name, package)
    provider_module = importlib.import_module(f"{package_name}.provider")
    registry = importlib.import_module("astrbot.core.provider.register")
    existing = {
        "openai_responses": object(),
        "openai_chat_completion": object(),
    }
    monkeypatch.setattr(registry, "provider_cls_map", dict(existing))
    monkeypatch.setattr(registry, "provider_registry", list(existing.values()))
    module_name = f"{package_name}.main"
    spec = importlib.util.spec_from_file_location(
        module_name, Path(__file__).resolve().parents[1] / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.StarTools, "get_data_dir", lambda name: tmp_path / name)
    try:
        yield SimpleNamespace(
            module=module,
            clients=provider_module,
            registry=registry,
            existing=existing,
        )
    finally:
        from astrbot.core.star.star import star_map, star_registry

        metadata = star_map.pop(module_name, None)
        star_registry[:] = [item for item in star_registry if item is not metadata]
        for name in list(sys.modules):
            if name.startswith(package_name + "."):
                sys.modules.pop(name, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_registration", [False, True])
async def test_import_activation_and_removal_leave_provider_registry_unchanged(
    lifecycle_env, legacy_registration
):
    env = lifecycle_env
    assert env.registry.provider_cls_map == env.existing
    assert env.registry.provider_registry == list(env.existing.values())
    if legacy_registration:
        # An external registration is never removed or replaced by this plugin.
        env.existing["openai_responses_image"] = object()
        env.registry.provider_cls_map.update(env.existing)
        env.registry.provider_registry[:] = env.existing.values()
    plugin = env.module.ResponsesImagePlugin(SimpleNamespace(register_web_api=Mock()))
    await plugin.initialize()
    library = plugin.library
    assert library is not None
    await plugin.initialize()
    assert plugin.library is library
    await plugin.terminate()
    await plugin.terminate()
    await plugin.initialize()
    assert env.registry.provider_cls_map == env.existing
    assert env.registry.provider_registry == list(env.existing.values())
    assert not hasattr(env.clients, "ImageResponsesProvider")
    assert not [
        cls
        for cls in vars(env.clients).values()
        if inspect.isclass(cls)
        and cls.__module__ == env.clients.__name__
        and issubclass(cls, Provider)
    ]


@pytest.mark.asyncio
async def test_deactivation_does_not_close_shared_models(lifecycle_env):
    env = lifecycle_env
    provider = object.__new__(env.module.ProviderOpenAIResponses)
    provider.client = SimpleNamespace(close=AsyncMock())
    provider.terminate = AsyncMock()
    manager = SimpleNamespace(
        inst_map={"shared-chat-and-drawing": provider},
        terminate_provider=AsyncMock(),
    )
    plugin = env.module.ResponsesImagePlugin(
        SimpleNamespace(provider_manager=manager, register_web_api=Mock())
    )
    await plugin.initialize()
    worker = env.clients.ResponsesImageClient(provider)
    await plugin.terminate()
    assert worker.provider is provider
    assert manager.inst_map == {"shared-chat-and-drawing": provider}
    provider.client.close.assert_not_awaited()
    provider.terminate.assert_not_awaited()
    manager.terminate_provider.assert_not_awaited()


@pytest.mark.asyncio
async def test_character_library_survives_plugin_reactivation(lifecycle_env, tmp_path):
    env = lifecycle_env
    plugin = env.module.ResponsesImagePlugin(SimpleNamespace(register_web_api=Mock()))
    await plugin.initialize()
    image = tmp_path / "character.png"
    PillowImage.new("RGB", (2, 2), "blue").save(image)
    profile = await plugin.library.save(
        "Character", ["Alias"], Image.fromFileSystem(image)
    )
    await plugin.terminate()
    replacement = env.module.ResponsesImagePlugin(
        SimpleNamespace(register_web_api=Mock())
    )
    await replacement.initialize()
    assert replacement.library.lookup("Alias") == profile
    assert Path(profile["image_path"]).is_file()
    assert env.registry.provider_cls_map == env.existing
