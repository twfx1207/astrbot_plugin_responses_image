"""Exercise searched-image network boundaries without contacting remote hosts."""

import importlib.util
import io
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image as PillowImage

_spec = importlib.util.spec_from_file_location(
    "responses_image_search_references_under_test",
    Path(__file__).resolve().parents[1] / "search_references.py",
)
assert _spec and _spec.loader
references = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = references
_spec.loader.exec_module(references)


class FakeResponse:
    """Provide a bounded asynchronous HTTP response double."""

    def __init__(
        self,
        chunks=(),
        *,
        status=200,
        location=None,
        content_length=None,
        content_type=None,
    ):
        """Prepare response bytes and redirect or length metadata.

        Args:
            chunks: Streamed body chunks.
            status: HTTP response status.
            location: Optional redirect destination.
            content_length: Optional advertised byte length.
            content_type: Optional declared content type.
        """
        self.chunks = chunks
        self.status = status
        self.headers = {"Location": location} if location is not None else {}
        if content_type is not None:
            self.headers["Content-Type"] = content_type
        self.content_length = content_length
        self.content = self

    async def __aenter__(self):
        """Enter the mocked HTTP response context.

        Returns:
            This response double.
        """
        return self

    async def __aexit__(self, *_args):
        """Exit the mocked HTTP response context.

        Args:
            *_args: Exception details from the context manager.
        """

    def raise_for_status(self):
        """Reject unsuccessful fake responses.

        Raises:
            RuntimeError: If the response is an HTTP error.
        """
        if self.status >= 400:
            raise RuntimeError("HTTP error")

    async def iter_chunked(self, _size):
        """Yield prepared body chunks.

        Args:
            _size: Requested chunk size.

        Yields:
            Prepared response bytes.
        """
        for chunk in self.chunks:
            yield chunk


@pytest.fixture
def network(monkeypatch):
    """Replace all connection creation with explicit response fixtures.

    Args:
        monkeypatch: Fixture restoring patched network constructors.

    Returns:
        Pending responses, observed requests and constructor mocks.
    """
    responses = []
    requests = []
    session = AsyncMock()
    session.__aenter__.return_value = session

    def get(url, **kwargs):
        """Record a request and consume one prepared response.

        Args:
            url: Requested URL.
            **kwargs: Connection options.

        Returns:
            A prepared response context manager.
        """
        requests.append((str(url), kwargs))
        return responses.pop(0)

    session.get = Mock(side_effect=get)
    session_constructor = Mock(return_value=session)
    connector_constructor = Mock()
    monkeypatch.setattr(references.aiohttp, "ClientSession", session_constructor)
    monkeypatch.setattr(references.aiohttp, "TCPConnector", connector_constructor)
    return SimpleNamespace(
        responses=responses,
        requests=requests,
        session=session_constructor,
        connector=connector_constructor,
    )


@pytest.fixture
def png_bytes():
    """Create actual image bytes for format validation.

    Returns:
        A small valid PNG image.
    """
    output = io.BytesIO()
    PillowImage.new("RGB", (2, 2), "blue").save(output, format="PNG")
    return output.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/picture.png",
        "http://a.LOCALHOST./picture.png",
        "http://127.0.0.1/picture.png",
        "http://10.1.2.3/picture.png",
        "http://169.254.169.254/latest/meta-data",
        "http://100.64.0.1/picture.png",
        "http://[::1]/picture.png",
        "http://[fc00::1]/picture.png",
        "http://[::ffff:127.0.0.1]/picture.png",
        "http://[64:ff9b::a00:1]/picture.png",
        "http://224.0.0.1/picture.png",
        "http://[ff02::1]/picture.png",
        "https://user:password@example.com/picture.png",
        "https://example.com:8443/picture.png",
        "https://example.com/\nimage.png",
    ],
)
async def test_unsafe_url_never_starts_request(url, tmp_path, network):
    with pytest.raises(ValueError):
        await references.download_reference_image(url, tmp_path)
    assert network.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address", ["127.0.0.1", "10.0.0.2", "::1", "224.0.0.1", "64:ff9b::a00:1"]
)
async def test_dns_rejects_mixed_public_and_private_addresses(address):
    resolver = references.PublicAddressResolver()
    resolver._resolver = SimpleNamespace(
        resolve=AsyncMock(return_value=[{"host": "8.8.8.8"}, {"host": address}]),
        close=AsyncMock(),
    )
    with pytest.raises(ValueError, match="public addresses"):
        await resolver.resolve("example.com", 443, socket.AF_UNSPEC)
    await resolver.close()
    resolver._resolver.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_dns_returns_the_checked_records_without_a_second_lookup():
    records = [{"host": "8.8.8.8"}, {"host": "2606:4700:4700::1111"}]
    resolver = references.PublicAddressResolver()
    resolver._resolver = SimpleNamespace(
        resolve=AsyncMock(return_value=records), close=AsyncMock()
    )
    assert await resolver.resolve("example.com", 443, socket.AF_UNSPEC) is records
    resolver._resolver.resolve.assert_awaited_once_with(
        "example.com", 443, socket.AF_UNSPEC
    )
    await resolver.close()


@pytest.mark.asyncio
async def test_valid_image_uses_guarded_resolver_and_verified_transport(
    tmp_path, network, png_bytes
):
    network.responses.append(FakeResponse([png_bytes]))
    path = await references.download_reference_image(
        "https://images.example.com/ref.png", tmp_path
    )
    assert path.parent == tmp_path
    assert path.suffix == ".png"
    assert path.read_bytes() == png_bytes
    connector_options = network.connector.call_args.kwargs
    assert isinstance(connector_options["resolver"], references.PublicAddressResolver)
    assert connector_options["use_dns_cache"] is False
    assert "ssl" not in connector_options
    assert network.session.call_args.kwargs["trust_env"] is False
    assert network.session.call_args.kwargs["auto_decompress"] is False
    options = network.requests[0][1]
    assert options["allow_redirects"] is False
    assert 0 < options["timeout"].total <= 20
    assert "ssl" not in options and "proxy" not in options


@pytest.mark.asyncio
async def test_redirect_to_private_address_is_rejected_before_second_request(
    tmp_path, network
):
    network.responses.append(
        FakeResponse(status=302, location="http://169.254.169.254/secret")
    )
    with pytest.raises(ValueError, match="private network"):
        await references.download_reference_image("https://example.com/ref", tmp_path)
    assert len(network.requests) == 1


@pytest.mark.asyncio
async def test_relative_redirect_keeps_validation_and_accepts_valid_image(
    tmp_path, network, png_bytes
):
    network.responses.extend(
        [FakeResponse(status=302, location="/actual.png"), FakeResponse([png_bytes])]
    )
    path = await references.download_reference_image(
        "https://example.com/ref", tmp_path
    )
    assert path.read_bytes() == png_bytes
    assert [url for url, _ in network.requests] == [
        "https://example.com/ref",
        "https://example.com/actual.png",
    ]


@pytest.mark.asyncio
async def test_redirect_loop_stops_after_three_redirects(tmp_path, network):
    network.responses.extend(
        FakeResponse(status=302, location="/again") for _ in range(4)
    )
    with pytest.raises(ValueError, match="redirects exceeded"):
        await references.download_reference_image("https://example.com/ref", tmp_path)
    assert len(network.requests) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("advertised", [True, False])
async def test_large_image_is_rejected_with_or_without_content_length(
    tmp_path, network, monkeypatch, advertised
):
    monkeypatch.setattr(references, "MAX_IMAGE_BYTES", 4)
    network.responses.append(
        FakeResponse([b"abc", b"def"], content_length=6 if advertised else None)
    )
    with pytest.raises(ValueError, match="exceeds"):
        await references.download_reference_image("https://example.com/ref", tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"<html>Access denied</html>", b"", b"GIF89a"])
async def test_invalid_image_bytes_are_rejected(tmp_path, network, body):
    network.responses.append(FakeResponse([body]))
    with pytest.raises(ValueError, match="valid image"):
        await references.download_reference_image("https://example.com/ref", tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_large_decoded_dimensions_are_rejected(
    tmp_path, network, png_bytes, monkeypatch
):
    monkeypatch.setattr(references, "MAX_IMAGE_PIXELS", 3)
    network.responses.append(FakeResponse([png_bytes]))
    with pytest.raises(ValueError, match="25 MP"):
        await references.download_reference_image("https://example.com/ref", tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_cited_pages_extract_real_images_and_spread_candidates_across_sources(
    network,
):
    first = b"""<html><title>Character profile</title>
        <meta property="og:image" content="/art/card.png">
        <img src="/logo.svg"><img width="16" height="16" src="/icon.png">
        <img src="/art/portrait.png" alt="Sherry official portrait">
        <img src="/art/portrait.png"><img data-src="/art/second.webp" src="/blank.gif">
        <script>var url = "https://invented.example/fake.png";</script>
        <a href="/not-an-image.png">Unrelated link</a></html>"""
    second = b'<meta name="twitter:image" content="//cdn.example/character.jpg">'
    network.responses.extend(
        [
            FakeResponse([first], content_type="text/html; charset=UTF-8"),
            FakeResponse([second], content_type="text/html"),
        ]
    )
    pages = [
        {
            "source_website_url": "https://official.example/profile",
            "caption": "Search title",
        },
        {"source_website_url": "https://wiki.example/profile", "caption": "Wiki title"},
    ]
    result = await references.resolve_reference_pages(pages, "Sherry")
    assert [item["image_url"] for item in result] == [
        "https://official.example/art/portrait.png",
        "https://cdn.example/character.jpg",
        "https://official.example/art/card.png",
    ]
    assert result[0]["source_website_url"] == pages[0]["source_website_url"]
    assert "Sherry official portrait" in result[0]["caption"]
    assert len(network.requests) == 2


@pytest.mark.asyncio
async def test_page_relative_images_use_final_redirect_url(network):
    network.responses.extend(
        [
            FakeResponse(status=302, location="/characters/sherry/index.html"),
            FakeResponse(
                [b'<img src="../portrait.png" alt="Sherry">'], content_type="text/html"
            ),
        ]
    )
    result = await references.resolve_reference_pages(
        [{"source_website_url": "https://official.example/old"}], "Sherry"
    )
    assert result[0]["image_url"] == "https://official.example/characters/portrait.png"
    assert (
        result[0]["source_website_url"]
        == "https://official.example/characters/sherry/index.html"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "page_url",
    ["http://127.0.0.1/private", "http://localhost/secret", "file:///private"],
)
async def test_cited_private_pages_are_never_fetched(network, page_url):
    result = await references.resolve_reference_pages(
        [{"source_website_url": page_url}], "Sherry"
    )
    assert result == [] and network.requests == []


@pytest.mark.asyncio
async def test_cited_page_redirects_cannot_reach_private_hosts(network):
    network.responses.append(
        FakeResponse(status=302, location="http://169.254.169.254/secret")
    )
    assert (
        await references.resolve_reference_pages(
            [{"source_website_url": "https://official.example/page"}], "Sherry"
        )
        == []
    )
    assert len(network.requests) == 1


@pytest.mark.asyncio
async def test_private_page_image_is_blocked_by_the_image_downloader(network, tmp_path):
    network.responses.append(
        FakeResponse(
            [b'<meta property="og:image" content="http://127.0.0.1/secret.png">']
        )
    )
    candidates = await references.resolve_reference_pages(
        [{"source_website_url": "https://official.example/page"}], "Sherry"
    )
    with pytest.raises(ValueError):
        await references.download_reference_image(candidates[0]["image_url"], tmp_path)
    assert len(network.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("advertised", [True, False])
async def test_page_body_size_is_bounded_and_other_sources_are_still_tried(
    network, monkeypatch, advertised
):
    monkeypatch.setattr(references, "MAX_PAGE_BYTES", 50)
    network.responses.extend(
        [
            FakeResponse([b"x" * 51], content_length=51 if advertised else None),
            FakeResponse([b'<img src="/ok.png">']),
        ]
    )
    result = await references.resolve_reference_pages(
        [
            {"source_website_url": "https://first.example/page"},
            {"source_website_url": "https://second.example/page"},
        ],
        "Sherry",
    )
    assert result[0]["image_url"] == "https://second.example/ok.png"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, media",
    [
        (b'<script>var img = "https://invented.example/x.png"</script>', "text/html"),
        (b'<a href="https://invented.example/x.png">photo</a>', "text/html"),
        (
            b'<img src="file:///secret"><img src="data:image/png;base64,secret">',
            "text/html",
        ),
        (b'{"image_url":"https://invented.example/x.png"}', "application/json"),
    ],
)
async def test_only_explicit_supported_html_image_attributes_are_used(
    network, body, media
):
    network.responses.append(FakeResponse([body], content_type=media))
    assert (
        await references.resolve_reference_pages(
            [{"source_website_url": "https://official.example/page"}], "Sherry"
        )
        == []
    )


@pytest.mark.asyncio
async def test_cited_image_resource_is_still_subject_to_image_validation(
    network, png_bytes, tmp_path
):
    network.responses.extend(
        [
            FakeResponse([png_bytes], content_type="image/png"),
            FakeResponse([png_bytes]),
        ]
    )
    candidates = await references.resolve_reference_pages(
        [{"source_website_url": "https://official.example/art.png"}], "Sherry"
    )
    path = await references.download_reference_image(
        candidates[0]["image_url"], tmp_path
    )
    assert path.read_bytes() == png_bytes
    assert list(tmp_path.iterdir()) == [path]
