"""Keep explicit event references and administrator-maintained character images."""

import asyncio
import base64
import binascii
import hashlib
import io
import json
import re
from pathlib import Path
from uuid import uuid4

import aiofiles
from astrbot.core.message import components as Comp
from PIL import Image as PillowImage

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
REFERENCE_EXTRA_KEY = "responses_image_references"


def collect_event_images(event) -> dict[str, dict]:
    """Register only this event's attachments and explicitly quoted images.

    Args:
        event: Incoming AstrBot message event.

    Returns:
        Stable event-local identifiers mapped to image components and sources.
    """
    cached = event.get_extra(REFERENCE_EXTRA_KEY)
    if cached is not None:
        return cached
    references = {}
    for component in event.get_messages():
        if isinstance(component, Comp.Image):
            references[f"ref_{uuid4().hex[:12]}"] = {
                "image": component,
                "source": "attachment",
            }
        elif isinstance(component, Comp.Reply):
            for quoted in component.chain or []:
                if isinstance(quoted, Comp.Image):
                    references[f"ref_{uuid4().hex[:12]}"] = {
                        "image": quoted,
                        "source": "reply",
                    }
    event.set_extra(REFERENCE_EXTRA_KEY, references)
    return references


def collect_generated_images(contexts: list[dict]) -> dict[str, dict]:
    """Recover the latest generated batch from linked tool and image messages.

    Only the current conversation snapshot is inspected. Cache paths establish
    provenance but are never opened; the saved image bytes are the edit source.

    Args:
        contexts: AstrBot's persisted messages for the current conversation.

    Returns:
        Event-local edit candidates from the latest complete generated batch.
        Missing provenance or incomplete image data yields no candidates.
    """
    calls = set()
    pending = set()
    latest = []
    for message in contexts[-100:]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            if pending:
                latest = []
            calls, pending = set(), set()
            tool_calls = message.get("tool_calls")
            for call in tool_calls if isinstance(tool_calls, list) else []:
                function = call.get("function") if isinstance(call, dict) else None
                if (
                    isinstance(function, dict)
                    and function.get("name") == "responses_draw_image"
                    and isinstance(call.get("id"), str)
                ):
                    calls.add(call["id"])
        elif (
            role == "tool"
            and isinstance(message.get("tool_call_id"), str)
            and message["tool_call_id"] in calls
        ):
            content = message.get("content")
            if isinstance(content, str):
                pending.update(
                    re.findall(
                        r"(?:^|\n\n)Image returned and cached at path='([^\r\n]{1,8192})'\. Review the image below\.",
                        content,
                    )
                )
        elif role == "user":
            if pending:
                latest = []
                parts = message.get("content")
                seen = set()
                if isinstance(parts, list):
                    for previous, part in zip(parts, parts[1:]):
                        if not isinstance(previous, dict) or not isinstance(part, dict):
                            continue
                        source = part.get("image_url")
                        if part.get("type") != "image_url" or not isinstance(
                            source, dict
                        ):
                            continue
                        image_id = source.get("id")
                        if (
                            not isinstance(image_id, str)
                            or image_id not in pending
                            or image_id in seen
                            or previous.get("type") != "text"
                            or previous.get("text")
                            != f"[Image from tool 'responses_draw_image', path='{image_id}']"
                        ):
                            continue
                        seen.add(image_id)
                        latest.append(source.get("url"))
                # Do not silently select an older image or part of a missing batch.
                if len(latest) != len(pending) or len(latest) > 4:
                    latest = []
            calls, pending = set(), set()
    if pending:
        return {}
    references = {}
    for index, url in enumerate(latest, 1):
        if not isinstance(url, str) or len(url) > 4 * ((MAX_IMAGE_BYTES + 2) // 3) + 64:
            return {}
        header, separator, encoded = url.partition(",")
        mime = {
            "data:image/png;base64": "PNG",
            "data:image/jpeg;base64": "JPEG",
            "data:image/webp;base64": "WEBP",
        }.get(header)
        if not separator or not mime:
            return {}
        try:
            data = base64.b64decode(encoded, validate=True)
            if len(data) > MAX_IMAGE_BYTES:
                return {}
            with PillowImage.open(io.BytesIO(data)) as picture:
                if (
                    picture.format != mime
                    or picture.width * picture.height > MAX_IMAGE_PIXELS
                ):
                    return {}
        except (
            ValueError,
            binascii.Error,
            OSError,
            PillowImage.DecompressionBombError,
        ):
            return {}
        references[f"prev_{uuid4().hex[:12]}"] = {
            "image": Comp.Image.fromBase64(encoded),
            "source": "latest_generated_image",
            "index": index,
        }
    return references


class CharacterLibrary:
    """Persist one verified-by-administrator reference image per character."""

    def __init__(self, data_dir: Path) -> None:
        """Load character metadata while enforcing the library path boundary.

        Args:
            data_dir: Dedicated persistent data directory owned by this plugin.

        Raises:
            ValueError: If existing character metadata is malformed or unsafe.
        """
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.data_dir / "characters.json"
        self._lock = asyncio.Lock()
        self._records: list[dict] = []
        if not self.metadata_path.exists():
            return
        try:
            if not self.metadata_path.resolve().is_relative_to(self.data_dir):
                raise ValueError("Character metadata must stay inside the library")
            records = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                raise ValueError("Character metadata must be a list")
            identities = set()
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError("Each character must be an object")
                name = record.get("name")
                aliases = record.get("aliases")
                filename = record.get("image")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("Each character needs a nonempty name")
                if not isinstance(aliases, list) or not all(
                    isinstance(alias, str) and alias.strip() for alias in aliases
                ):
                    raise ValueError("Character aliases must be nonempty strings")
                if (
                    not isinstance(filename, str)
                    or not filename
                    or any(part in filename for part in ("/", "\\", ":", "\x00"))
                    or filename in {".", ".."}
                    or not (self.data_dir / filename)
                    .resolve()
                    .is_relative_to(self.data_dir)
                ):
                    raise ValueError(
                        "Character image must be a file inside the library"
                    )
                keys = {
                    " ".join(value.split()).casefold() for value in [name, *aliases]
                }
                if keys & identities:
                    raise ValueError("Character names and aliases must be unique")
                identities.update(keys)
                self._records.append(
                    {"name": name, "aliases": list(aliases), "image": filename}
                )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Cannot read character library metadata") from exc

    def list_names(self) -> list[str]:
        """Return the canonical names available to the current process.

        Returns:
            Character names sorted without regard to letter case.
        """
        return sorted((record["name"] for record in self._records), key=str.casefold)

    def lookup(self, name: str) -> dict | None:
        """Resolve a canonical name or alias without exposing mutable metadata.

        Args:
            name: Canonical name or an administrator-declared alias.

        Returns:
            Character name, aliases and absolute reference path, or None.

        Raises:
            ValueError: If a stored reference no longer stays inside the library.
        """
        key = " ".join(name.split()).casefold()
        for record in self._records:
            if key in {
                " ".join(value.split()).casefold()
                for value in [record["name"], *record["aliases"]]
            }:
                image_path = (self.data_dir / record["image"]).resolve()
                if not image_path.is_relative_to(self.data_dir):
                    raise ValueError("Character image must stay inside the library")
                return {
                    "name": record["name"],
                    "aliases": list(record["aliases"]),
                    "image_path": str(image_path),
                }
        return None

    async def save(self, name: str, aliases: list[str], image: Comp.Image) -> dict:
        """Validate and atomically store an administrator-selected character image.

        Args:
            name: Canonical character name, including the work when ambiguous.
            aliases: Alternative names for this same character and appearance.
            image: Explicit incoming image chosen by the administrator.

        Returns:
            The saved character name, aliases and absolute reference path.

        Raises:
            ValueError: If the image, names or aliases are invalid or conflicting.
            OSError: If the library cannot persist the reference or metadata.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("A character name is required")
        name = " ".join(name.split())
        if len(name) > 200 or not isinstance(aliases, list):
            raise ValueError("Invalid character name or alias list")
        normalized_aliases = []
        own_keys = {name.casefold()}
        for alias in aliases:
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError("Character aliases must be nonempty strings")
            alias = " ".join(alias.split())
            if len(alias) > 200:
                raise ValueError("Character aliases must not exceed 200 characters")
            if alias.casefold() not in own_keys:
                normalized_aliases.append(alias)
                own_keys.add(alias.casefold())

        source = Path(await image.convert_to_file_path())
        async with aiofiles.open(source, "rb") as handle:
            contents = await handle.read(MAX_IMAGE_BYTES + 1)
        if not contents or len(contents) > MAX_IMAGE_BYTES:
            raise ValueError("Character reference must be between 1 byte and 20 MiB")
        try:
            with PillowImage.open(io.BytesIO(contents)) as parsed:
                suffix = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(
                    parsed.format
                )
                if not suffix:
                    raise ValueError("Character reference must be PNG, JPEG or WebP")
                if parsed.width * parsed.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Character reference exceeds 25 million pixels")
                await asyncio.to_thread(parsed.load)
        except (OSError, PillowImage.DecompressionBombError) as exc:
            raise ValueError("Character reference is not a valid image") from exc

        filename = hashlib.sha256(contents).hexdigest() + suffix
        destination = self.data_dir / filename
        async with self._lock:
            for record in self._records:
                existing_name = " ".join(record["name"].split()).casefold()
                if existing_name == name.casefold():
                    continue
                existing_keys = {
                    " ".join(value.split()).casefold()
                    for value in [record["name"], *record["aliases"]]
                }
                if own_keys & existing_keys:
                    raise ValueError(
                        "A name or alias already belongs to another character"
                    )
            records = [
                record
                for record in self._records
                if " ".join(record["name"].split()).casefold() != name.casefold()
            ]
            records.append(
                {"name": name, "aliases": normalized_aliases, "image": filename}
            )
            image_temp = self.data_dir / f".image-{uuid4().hex}.tmp"
            metadata_temp = self.data_dir / f".characters-{uuid4().hex}.tmp"
            try:
                async with aiofiles.open(image_temp, "xb") as handle:
                    await handle.write(contents)
                # Replacing the directory entry also avoids following a stale symlink.
                image_temp.replace(destination)
                async with aiofiles.open(
                    metadata_temp, "x", encoding="utf-8"
                ) as handle:
                    await handle.write(
                        json.dumps(records, ensure_ascii=False, indent=2)
                    )
                metadata_temp.replace(self.metadata_path)
                self._records = records
            finally:
                image_temp.unlink(missing_ok=True)
                metadata_temp.unlink(missing_ok=True)
        return {
            "name": name,
            "aliases": list(normalized_aliases),
            "image_path": str(destination),
        }
