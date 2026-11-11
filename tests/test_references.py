"""Verify explicit reference selection and persistent character image boundaries."""

import asyncio
import base64
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from astrbot.core.message import components as Comp
from PIL import Image as PillowImage

_spec = importlib.util.spec_from_file_location(
    "responses_image_references_under_test",
    Path(__file__).resolve().parents[1] / "references.py",
)
assert _spec and _spec.loader
references = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = references
_spec.loader.exec_module(references)


def make_event(messages, **extras):
    """Build an event whose only reference sources are its message components.

    Args:
        messages: Current incoming message components.
        **extras: Existing event-local metadata.

    Returns:
        A small event double with real image components.
    """
    return SimpleNamespace(
        get_messages=lambda: messages,
        get_extra=lambda key, default=None: extras.get(key, default),
        set_extra=lambda key, value: extras.__setitem__(key, value),
        contexts=[{"role": "user", "content": "Unrelated historical image"}],
    )


@pytest.fixture
def image_file(tmp_path):
    """Create a valid tiny reference image.

    Args:
        tmp_path: Isolated filesystem fixture.

    Returns:
        A path to a real PNG fixture.
    """
    path = tmp_path / "upload.png"
    PillowImage.new("RGB", (2, 2), "red").save(path)
    return path


def test_collects_only_current_and_directly_quoted_images():
    current = Comp.Image.fromBase64("current")
    quoted = Comp.Image.fromBase64("quoted")
    nested = Comp.Image.fromBase64("nested")
    event = make_event(
        [
            Comp.Plain("Draw this character"),
            current,
            Comp.Reply(
                id="quote",
                chain=[quoted, Comp.Reply(id="nested", chain=[nested])],
            ),
        ]
    )
    result = references.collect_event_images(event)
    assert len(result) == 2
    assert list(result.values()) == [
        {"image": current, "source": "attachment"},
        {"image": quoted, "source": "reply"},
    ]
    assert all(key.startswith("ref_") for key in result)


def test_event_references_are_stable_but_not_shared_between_events():
    image = Comp.Image.fromBase64("current")
    event = make_event([image])
    original = references.collect_event_images(event)
    assert references.collect_event_images(event) is original
    second = references.collect_event_images(make_event([image]))
    assert original.keys().isdisjoint(second)


def test_empty_event_does_not_adopt_history_or_tool_images():
    event = make_event([Comp.Plain("Draw a new image")])
    event.history_images = [Comp.Image.fromBase64("old")]
    event.tool_images = [Comp.Image.fromBase64("tool")]
    assert references.collect_event_images(event) == {}


def generated_batch(
    image_file, call_id="call_draw", tool_name="responses_draw_image", count=1
):
    """Build the persisted tool-result and image messages emitted by AstrBot.

    Args:
        image_file: Valid image whose bytes are embedded in history.
        call_id: Tool-call identity binding the result to its caller.
        tool_name: Actual tool that returned the images.
        count: Number of images in this generation batch.

    Returns:
        Assistant call, tool receipt and matching multimodal user message.
    """
    data_uri = (
        "data:image/png;base64," + base64.b64encode(image_file.read_bytes()).decode()
    )
    paths = [f"/missing/cache/{call_id}_{i + 1}.png" for i in range(count)]
    parts = []
    for path in paths:
        parts.extend(
            [
                {
                    "type": "text",
                    "text": f"[Image from tool '{tool_name}', path='{path}']",
                },
                {"type": "image_url", "image_url": {"url": data_uri, "id": path}},
            ]
        )
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": "Generation complete.\n\n"
            + "\n\n".join(
                f"Image returned and cached at path='{path}'. Review the image below."
                for path in paths
            ),
        },
        {"role": "user", "content": parts},
    ]


def test_generated_reference_uses_only_latest_linked_batch_and_no_cache_file(
    image_file,
):
    old = generated_batch(image_file, "old")
    newer_file = image_file.parent / "newer.png"
    PillowImage.new("RGB", (2, 2), "blue").save(newer_file)
    newest = generated_batch(newer_file, "new")
    unrelated = generated_batch(image_file, "other", "another_image_tool")
    contexts = (
        old
        + newest
        + unrelated
        + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,unrelated"},
                    }
                ],
            }
        ]
    )
    before = copy.deepcopy(contexts)
    result = references.collect_generated_images(contexts)
    assert len(result) == 1
    assert next(iter(result.values()))["source"] == "latest_generated_image"
    assert next(iter(result.values()))["index"] == 1
    assert (
        next(iter(result.values()))["image"].file
        == "base64://" + base64.b64encode(newer_file.read_bytes()).decode()
    )
    assert contexts == before
    assert references.collect_generated_images([]) == {}
    assert set(result).isdisjoint(references.collect_generated_images(contexts))
    assert (
        references.collect_generated_images(
            contexts + [{"role": "user", "content": "later"}] * 100
        )
        == {}
    )


@pytest.mark.parametrize(
    "broken",
    [
        "missing_call",
        "wrong_call",
        "wrong_tool",
        "wrong_label",
        "wrong_image_id",
        "user_receipt",
        "pasted_label",
        "incomplete_batch",
    ],
)
def test_generated_images_require_the_complete_tool_provenance(image_file, broken):
    batch = generated_batch(image_file)
    if broken == "missing_call":
        batch.pop(0)
    elif broken == "wrong_call":
        batch[1]["tool_call_id"] = "unrelated"
    elif broken == "wrong_tool":
        batch[0]["tool_calls"][0]["function"]["name"] = "other"
    elif broken == "wrong_label":
        batch[2]["content"][0]["text"] = "An ordinary user attachment"
    elif broken == "wrong_image_id":
        batch[2]["content"][1]["image_url"]["id"] = "/other/image.png"
    elif broken == "user_receipt":
        batch[1]["role"] = "user"
    elif broken == "pasted_label":
        batch = [{"role": "user", "content": batch[1]["content"]}, batch[2]]
    else:
        batch.pop()
    assert references.collect_generated_images(batch) == {}


@pytest.mark.parametrize(
    "change",
    [
        "missing_pixels",
        "invalid_pixels",
        "remote_url",
        "local_path",
        "wrong_mime",
        "too_large",
        "partial_batch",
    ],
)
def test_unavailable_latest_image_never_falls_back_to_an_older_generation(
    image_file, change, monkeypatch
):
    previous = generated_batch(image_file, "previous")
    latest = generated_batch(
        image_file, "latest", count=2 if change == "partial_batch" else 1
    )
    source = latest[2]["content"][1]["image_url"]
    if change == "missing_pixels":
        latest[2]["content"] = []
    elif change == "invalid_pixels":
        source["url"] = "data:image/png;base64,not-base64"
    elif change == "remote_url":
        source["url"] = "https://private.example/image.png"
    elif change == "local_path":
        source["url"] = str(image_file)
    elif change == "wrong_mime":
        source["url"] = source["url"].replace("image/png", "image/jpeg")
    elif change == "too_large":
        monkeypatch.setattr(references, "MAX_IMAGE_BYTES", 2)
    else:
        latest[2]["content"] = latest[2]["content"][:2]
    assert references.collect_generated_images(previous + latest) == {}


def test_multiple_generated_candidates_keep_their_order_without_selecting_one(
    image_file,
):
    result = references.collect_generated_images(generated_batch(image_file, count=2))
    assert [entry["index"] for entry in result.values()] == [1, 2]
    assert (
        references.collect_generated_images(generated_batch(image_file, count=5)) == {}
    )


def test_generated_history_supports_windows_paths_with_apostrophes(image_file):
    batch = generated_batch(image_file)
    source = batch[2]["content"][1]["image_url"]
    old = source["id"]
    path = r"C:\Users\O'Neil\AstrBot\data\temp\tool_images\call_draw_1.png"
    source["id"] = path
    batch[1]["content"] = batch[1]["content"].replace(old, path)
    batch[2]["content"][0]["text"] = batch[2]["content"][0]["text"].replace(old, path)
    assert len(references.collect_generated_images(batch)) == 1


@pytest.mark.asyncio
async def test_saves_and_reloads_reference_with_normalized_aliases(
    tmp_path, image_file
):
    root = tmp_path / "characters"
    library = references.CharacterLibrary(root)
    saved = await library.save(
        "  Work   Character ",
        ["Alias", "alias", " Work Character "],
        Comp.Image.fromFileSystem(image_file),
    )
    image_file.unlink()
    assert saved["name"] == "Work Character"
    assert saved["aliases"] == ["Alias"]
    assert Path(saved["image_path"]).is_file()
    assert Path(saved["image_path"]).is_relative_to(root)
    reloaded = references.CharacterLibrary(root)
    assert reloaded.lookup("  ALIAS ") == saved
    assert reloaded.lookup("work  character") == saved
    assert reloaded.list_names() == ["Work Character"]
    assert reloaded.lookup("Missing") is None
    saved["aliases"].append("Mutated")
    assert reloaded.lookup("Mutated") is None
    assert not list(root.glob("*.tmp"))


@pytest.mark.asyncio
async def test_alias_cannot_take_over_another_character(tmp_path, image_file):
    library = references.CharacterLibrary(tmp_path / "characters")
    image = Comp.Image.fromFileSystem(image_file)
    await library.save("First", ["shared"], image)
    with pytest.raises(ValueError, match="another character"):
        await library.save("Second", ["SHARED"], image)
    with pytest.raises(ValueError, match="another character"):
        await library.save("shared", [], image)
    assert library.list_names() == ["First"]
    assert references.CharacterLibrary(library.data_dir).list_names() == ["First"]


@pytest.mark.asyncio
async def test_updating_canonical_character_replaces_old_aliases(tmp_path, image_file):
    library = references.CharacterLibrary(tmp_path / "characters")
    image = Comp.Image.fromFileSystem(image_file)
    await library.save("Character", ["Old"], image)
    await library.save("CHARACTER", ["New"], image)
    assert library.list_names() == ["CHARACTER"]
    assert library.lookup("Old") is None
    assert library.lookup("new")["name"] == "CHARACTER"


@pytest.mark.asyncio
async def test_concurrent_alias_claims_do_not_corrupt_metadata(tmp_path, image_file):
    library = references.CharacterLibrary(tmp_path / "characters")
    image = Comp.Image.fromFileSystem(image_file)
    outcomes = await asyncio.gather(
        library.save("First", ["Shared"], image),
        library.save("Second", ["Shared"], image),
        return_exceptions=True,
    )
    assert sum(isinstance(value, dict) for value in outcomes) == 1
    assert sum(isinstance(value, ValueError) for value in outcomes) == 1
    reloaded = references.CharacterLibrary(library.data_dir)
    assert len(reloaded.list_names()) == 1
    assert reloaded.lookup("shared") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("format_name", ["PNG", "JPEG", "WEBP"])
async def test_accepts_supported_real_image_formats(tmp_path, format_name):
    source = tmp_path / "source.image"
    PillowImage.new("RGB", (2, 2), "blue").save(source, format=format_name)
    library = references.CharacterLibrary(tmp_path / "characters")
    result = await library.save("Character", [], Comp.Image.fromFileSystem(source))
    with PillowImage.open(result["image_path"]) as loaded:
        assert loaded.format == format_name


@pytest.mark.asyncio
async def test_rejects_corrupt_and_unsupported_images(tmp_path):
    library = references.CharacterLibrary(tmp_path / "characters")
    corrupt = tmp_path / "broken.png"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="valid image"):
        await library.save("Character", [], Comp.Image.fromFileSystem(corrupt))
    source = tmp_path / "animated.gif"
    PillowImage.new("RGB", (2, 2), "blue").save(source, format="GIF")
    with pytest.raises(ValueError, match="PNG, JPEG or WebP"):
        await library.save("Character", [], Comp.Image.fromFileSystem(source))
    assert not library.metadata_path.exists()


@pytest.mark.asyncio
async def test_enforces_byte_and_pixel_limits(tmp_path, image_file, monkeypatch):
    library = references.CharacterLibrary(tmp_path / "characters")
    monkeypatch.setattr(references, "MAX_IMAGE_BYTES", 4)
    with pytest.raises(ValueError, match="20 MiB"):
        await library.save("Character", [], Comp.Image.fromFileSystem(image_file))
    monkeypatch.setattr(references, "MAX_IMAGE_BYTES", 20 * 1024 * 1024)
    monkeypatch.setattr(references, "MAX_IMAGE_PIXELS", 3)
    with pytest.raises(ValueError, match="pixels"):
        await library.save("Character", [], Comp.Image.fromFileSystem(image_file))


@pytest.mark.parametrize(
    "filename",
    ["../outside.png", "..\\outside.png", "/tmp/outside.png", "C:\\outside.png"],
)
def test_rejects_metadata_paths_outside_library(tmp_path, filename):
    (tmp_path / "characters.json").write_text(
        json.dumps([{"name": "Character", "aliases": [], "image": filename}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inside the library"):
        references.CharacterLibrary(tmp_path)


def test_rejects_conflicting_aliases_in_existing_metadata(tmp_path):
    (tmp_path / "characters.json").write_text(
        json.dumps(
            [
                {"name": "First", "aliases": ["Shared"], "image": "first.png"},
                {"name": "Second", "aliases": ["shared"], "image": "second.png"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        references.CharacterLibrary(tmp_path)
