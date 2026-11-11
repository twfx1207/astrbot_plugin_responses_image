"""Retrieve source-linked character images using AstrBot's Tavily settings."""

import json
from urllib.parse import urlsplit

import aiohttp


async def search_tavily_references(query: str, config: dict) -> list[dict[str, str]]:
    """Search once for images attached to actual Tavily source results.

    Args:
        query: A focused character reference query, at most 1500 characters.
        config: The current AstrBot configuration, including provider_settings.

    Returns:
        Up to three distinct image URLs with their source pages and captions.

    Raises:
        ValueError: If the query or configured Tavily API keys are invalid.
        RuntimeError: If the service fails or returns no source-linked images.
        aiohttp.ClientError: If the HTTP request fails.
        TimeoutError: If the request exceeds its time limit.
    """
    if not isinstance(query, str) or not query.strip() or len(query) > 1500:
        raise ValueError("Tavily reference queries must contain 1 to 1500 characters.")
    settings = config.get("provider_settings", {}) if isinstance(config, dict) else {}
    keys = (
        settings.get("websearch_tavily_key", []) if isinstance(settings, dict) else []
    )
    key = (
        next(
            (
                value.strip()
                for value in keys
                if isinstance(value, str) and value.strip()
            ),
            "",
        )
        if isinstance(keys, list)
        else ""
    )
    if not key:
        raise ValueError("Configure a Tavily API key in AstrBot provider settings.")

    # Keep this separate from AstrBot's text-only search result normalization.
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30), trust_env=False
    ) as session:
        async with session.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "query": query.strip(),
                "search_depth": "basic",
                "max_results": 3,
                "include_images": True,
                "include_image_descriptions": True,
                "include_answer": False,
                "include_raw_content": False,
                "auto_parameters": False,
            },
            allow_redirects=False,
        ) as response:
            if response.status != 200:
                # Do not expose API response bodies or credentials in tool errors.
                raise RuntimeError(
                    f"Tavily reference search returned HTTP {response.status}."
                )
            maximum = 2 * 1024 * 1024
            if (
                response.content_length is not None
                and response.content_length > maximum
            ):
                raise RuntimeError("Tavily reference response exceeds 2 MiB.")
            body = bytearray()
            async for chunk in response.content.iter_chunked(65536):
                body.extend(chunk)
                if len(body) > maximum:
                    raise RuntimeError("Tavily reference response exceeds 2 MiB.")
            try:
                data = json.loads(body)
            except (ValueError, UnicodeError) as exc:
                raise RuntimeError("Tavily returned an invalid JSON response.") from exc

    results = data.get("results", []) if isinstance(data, dict) else []
    if not isinstance(results, list):
        raise RuntimeError("Tavily returned an invalid result list.")
    references = []
    seen = set()
    for result in results[:3]:
        if not isinstance(result, dict):
            continue
        source = result.get("url")
        images = result.get("images", [])
        if not isinstance(source, str) or not isinstance(images, list):
            continue
        # Top-level images lack a reliable source-page association and are ignored.
        for image in images[:20]:
            if not isinstance(image, dict):
                continue
            image_url = image.get("url")
            if not isinstance(image_url, str) or image_url in seen:
                continue
            if any(
                not value
                or len(value) > 8192
                or any(ord(char) <= 32 or ord(char) == 127 for char in value)
                for value in (source, image_url)
            ):
                continue
            try:
                urls = [urlsplit(source), urlsplit(image_url)]
                if any(
                    url.scheme not in {"https", "http"}
                    or not url.hostname
                    or url.username is not None
                    or url.password is not None
                    for url in urls
                ):
                    continue
                for url in urls:
                    url.port
            except ValueError:
                continue
            description = image.get("description", "")
            title = result.get("title", "")
            content = result.get("content", "")
            caption = "\n".join(
                f"{label}: {value[:limit]}"
                for label, value, limit in (
                    ("Image description", description, 1000),
                    ("Source title", title, 300),
                    ("Source excerpt", content, 1500),
                )
                if isinstance(value, str) and value.strip()
            )
            seen.add(image_url)
            references.append(
                {
                    "image_url": image_url,
                    "source_website_url": source,
                    "caption": caption,
                }
            )
            if len(references) == 3:
                return references
    if not references:
        raise RuntimeError("Tavily returned no images with an associated source page.")
    return references
