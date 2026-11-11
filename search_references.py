"""Download untrusted search images without permitting private network access."""

import asyncio
import hashlib
import io
import ipaddress
import socket
from html.parser import HTMLParser
from pathlib import Path

import aiofiles
import aiohttp
from astrbot import logger
from PIL import Image as PillowImage
from yarl import URL

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_PAGE_BYTES = 2 * 1024 * 1024


class PublicAddressResolver(aiohttp.abc.AbstractResolver):
    """Validate the same DNS addresses that the HTTP connector will use."""

    def __init__(self) -> None:
        """Create the system DNS resolver used for public image hosts."""
        self._resolver = aiohttp.resolver.ThreadedResolver()

    async def resolve(self, host, port=0, family=socket.AF_INET):
        """Reject DNS responses containing any non-public address.

        Args:
            host: Requested server hostname.
            port: Destination port.
            family: Requested address family.

        Returns:
            Validated connector address records from the system resolver.

        Raises:
            ValueError: If any address is not globally routable.
        """
        addresses = await self._resolver.resolve(host, port, family)
        if not addresses or any(
            not (address := ipaddress.ip_address(item["host"])).is_global
            or address.is_multicast
            or address.is_reserved
            for item in addresses
        ):
            raise ValueError("Reference host must resolve only to public addresses")
        return addresses

    async def close(self) -> None:
        """Release the underlying DNS resolver."""
        await self._resolver.close()


async def _fetch_reference_bytes(
    url: str, maximum: int, accept: str
) -> tuple[bytes, str, str]:
    """Fetch a bounded public resource with the same checks for pages and images.

    Args:
        url: Untrusted HTTP or HTTPS resource URL.
        maximum: Maximum allowed response body length in bytes.
        accept: Accepted media types advertised to the server.

    Returns:
        Response bytes, final validated URL and declared content type.

    Raises:
        ValueError: If a URL, resolved address, redirect, or body size is unsafe.
        aiohttp.ClientError: If the remote server or TLS connection fails.
        TimeoutError: If the total download deadline expires.
    """
    if not isinstance(url, str) or not url or len(url) > 8192:
        raise ValueError("Reference URL must be a bounded nonempty string")
    if any(ord(character) <= 32 or ord(character) == 127 for character in url):
        raise ValueError("Reference URL cannot contain whitespace or controls")
    target = URL(url)
    resolver = PublicAddressResolver()
    deadline = asyncio.get_running_loop().time() + 20
    try:
        connector = aiohttp.TCPConnector(resolver=resolver, use_dns_cache=False)
        async with aiohttp.ClientSession(
            connector=connector,
            trust_env=False,
            auto_decompress=False,
            headers={
                "Accept": accept,
                "Accept-Encoding": "identity",
            },
        ) as session:
            for redirect_count in range(4):
                host = target.host
                if (
                    target.scheme not in {"http", "https"}
                    or not host
                    or target.user is not None
                    or target.password is not None
                    or target.port not in {80, 443}
                    or "%" in host
                    or host.rstrip(".").casefold() == "localhost"
                    or host.rstrip(".").casefold().endswith(".localhost")
                ):
                    raise ValueError("Reference URL must use a public HTTP(S) host")
                try:
                    literal_address = ipaddress.ip_address(host)
                except ValueError:
                    pass
                else:
                    # aiohttp bypasses the resolver for IP literals.
                    if (
                        not literal_address.is_global
                        or literal_address.is_multicast
                        or literal_address.is_reserved
                    ):
                        raise ValueError(
                            "Reference URL cannot address a private network"
                        )
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("Reference image download timed out")
                async with session.get(
                    target,
                    allow_redirects=False,
                    timeout=aiohttp.ClientTimeout(total=remaining),
                ) as response:
                    if response.status in {301, 302, 303, 307, 308}:
                        location = response.headers.get("Location", "")
                        if redirect_count == 3 or not location or len(location) > 8192:
                            raise ValueError(
                                "Reference image redirects exceeded the limit"
                            )
                        if any(ord(c) <= 32 or ord(c) == 127 for c in location):
                            raise ValueError(
                                "Reference redirect contains invalid characters"
                            )
                        target = target.join(URL(location))
                        continue
                    response.raise_for_status()
                    if response.status != 200:
                        raise ValueError(
                            "Reference server did not return a complete image"
                        )
                    if (
                        response.content_length is not None
                        and response.content_length > maximum
                    ):
                        raise ValueError(f"Reference response exceeds {maximum} bytes")
                    data = bytearray()
                    async for chunk in response.content.iter_chunked(64 * 1024):
                        if len(data) + len(chunk) > maximum:
                            raise ValueError(
                                f"Reference response exceeds {maximum} bytes"
                            )
                        data.extend(chunk)
                    return (
                        bytes(data),
                        str(target),
                        response.headers.get("Content-Type", ""),
                    )
    finally:
        await resolver.close()
    raise ValueError("Reference resource was not returned")


async def download_reference_image(url: str, directory: Path) -> Path:
    """Fetch and validate one image obtained from an untrusted search result.

    Args:
        url: An HTTP or HTTPS image URL returned by search or a cited page.
        directory: Caller-owned temporary directory removed after generation.

    Returns:
        A local PNG, JPEG, or WebP path containing the downloaded bytes.

    Raises:
        ValueError: If the address, response size, or image format is invalid.
        aiohttp.ClientError: If the remote server or TLS connection fails.
        TimeoutError: If the download deadline expires.
        OSError: If the local image cannot be saved.
    """
    data, _, _ = await _fetch_reference_bytes(
        url, MAX_IMAGE_BYTES, "image/png,image/jpeg,image/webp"
    )

    try:
        with PillowImage.open(io.BytesIO(data)) as image:
            extension = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(
                image.format
            )
            if extension is None or image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("Reference must be PNG, JPEG, or WebP within 25 MP")
            await asyncio.to_thread(image.load)
    except (OSError, PillowImage.DecompressionBombError) as exc:
        raise ValueError("Reference response does not contain a valid image") from exc
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / (hashlib.sha256(data).hexdigest() + extension)
    async with aiofiles.open(destination, "wb") as output:
        await output.write(data)
    return destination


class _ReferencePageParser(HTMLParser):
    """Collect explicit image attributes from static HTML without running scripts."""

    def __init__(self, page_url: str, character_name: str) -> None:
        """Initialize bounded candidates for one cited page.

        Args:
            page_url: Final validated page URL used to resolve relative images.
            character_name: Requested name used only to rank explicit alt text.
        """
        super().__init__(convert_charrefs=True)
        self.page_url = URL(page_url)
        self.character_name = " ".join(character_name.split()).casefold()
        self.images: list[tuple[int, str, str]] = []
        self.title = ""
        self.description = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Read image metadata and ordinary image elements from HTML.

        Args:
            tag: Parsed element name.
            attrs: Parsed element attributes; treated only as untrusted data.
        """
        values = dict(attrs)
        candidate = ""
        caption = ""
        priority = 2
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            key = (values.get("property") or values.get("name") or "").casefold()
            value = values.get("content") or ""
            if key in {
                "og:image",
                "og:image:url",
                "og:image:secure_url",
                "twitter:image",
                "twitter:image:src",
            }:
                candidate, priority = value, 1
            elif key == "og:title":
                self.title = value[:300]
            elif key in {"og:description", "description"}:
                self.description = value[:1000]
        elif tag == "link" and "image_src" in (values.get("rel") or "").split():
            candidate, priority = values.get("href") or "", 1
        elif tag == "img":
            for dimension in ("width", "height"):
                value = values.get(dimension) or ""
                if (
                    value.isascii()
                    and value.isdecimal()
                    and len(value) < 6
                    and int(value) < 64
                ):
                    return
            candidate = (
                values.get("data-src")
                or values.get("data-original")
                or values.get("src")
                or ""
            )
            caption = (values.get("alt") or values.get("title") or "")[:300]
            if (
                self.character_name
                and self.character_name in " ".join(caption.split()).casefold()
            ):
                priority = 0
        if not candidate or len(candidate) > 8192:
            return
        if any(ord(char) <= 32 or ord(char) == 127 for char in candidate):
            return
        try:
            image_url = self.page_url.join(URL(candidate)).with_fragment(None)
            if (
                image_url.scheme not in {"http", "https"}
                or not image_url.host
                or image_url.user is not None
                or image_url.password is not None
                or image_url.port not in {80, 443}
                or image_url.path.casefold().endswith((".svg", ".ico", ".gif"))
            ):
                return
        except (ValueError, UnicodeError):
            return
        if any(
            existing[1] == str(image_url) and existing[0] <= priority
            for existing in self.images
        ):
            return
        self.images = [
            existing for existing in self.images if existing[1] != str(image_url)
        ]
        self.images.append((priority, str(image_url), caption))
        # Keep the strongest candidates even if a page has a long navigation bar.
        self.images.sort(key=lambda item: item[0])
        del self.images[24:]

    def handle_data(self, data: str) -> None:
        """Keep only a bounded page title, excluding body text and scripts.

        Args:
            data: Text content from the parser.
        """
        if self._in_title:
            self.title = (self.title + data)[:300]

    def handle_endtag(self, tag: str) -> None:
        """Stop collecting title text when its element closes.

        Args:
            tag: Closing element name.
        """
        if tag == "title":
            self._in_title = False


async def resolve_reference_pages(
    pages: list[dict[str, str]], character_name: str
) -> list[dict[str, str]]:
    """Extract up to three image candidates from actual native search citations.

    Args:
        pages: Up to three page records emitted by the native citation parser.
        character_name: Requested character name for ranking explicit alt text.

    Returns:
        Source-linked image candidates, pending image validation and visual checks.
        An empty list means no cited page yielded a usable static image address.
    """
    per_page = []
    for page in pages[:3]:
        try:
            data, final_url, content_type = await _fetch_reference_bytes(
                page["source_website_url"],
                MAX_PAGE_BYTES,
                "text/html,application/xhtml+xml,image/png,image/jpeg,image/webp",
            )
            media_type = content_type.split(";", 1)[0].strip().lower()
            if media_type in {"image/png", "image/jpeg", "image/webp"}:
                # A citation may point directly at an image; the next stage decodes it.
                per_page.append(
                    [
                        {
                            "image_url": final_url,
                            "source_website_url": final_url,
                            "caption": page.get("caption", "")[:300],
                        }
                    ]
                )
                continue
            if media_type not in {"", "text/html", "application/xhtml+xml"}:
                raise ValueError("Cited resource is not HTML or a supported image")
            parser = _ReferencePageParser(final_url, character_name)
            parser.feed(data.decode("utf-8", errors="replace"))
            parser.close()
            found = []
            seen = set()
            for _, image_url, caption in parser.images:
                if image_url in seen:
                    continue
                seen.add(image_url)
                found.append(
                    {
                        "image_url": image_url,
                        "source_website_url": final_url,
                        "caption": (
                            f"Citation title: {page.get('caption', '')[:300]}\n"
                            f"Page title: {parser.title}\n"
                            f"Image description: {caption}\n"
                            f"Page description: {parser.description}"
                        )[:2000],
                    }
                )
                if len(found) == 3:
                    break
            per_page.append(found)
            logger.debug("Cited reference page parsed: image_candidates=%d", len(found))
        except Exception as exc:
            # URLs, page bodies and exception bodies must stay out of diagnostics.
            logger.debug("Cited reference page unavailable: %s", type(exc).__name__)
    candidates = []
    seen_images = set()
    # Give each cited page a chance before trying more images from a single page.
    for index in range(3):
        for found in per_page:
            if index < len(found) and found[index]["image_url"] not in seen_images:
                candidate = found[index]
                seen_images.add(candidate["image_url"])
                candidates.append(candidate)
                if len(candidates) == 3:
                    return candidates
    return candidates
