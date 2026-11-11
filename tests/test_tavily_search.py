"""Test bounded Tavily searches without credentials or network access."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def tavily_env(monkeypatch):
    """Load the adapter with a controllable HTTP session.

    Args:
        monkeypatch: Fixture restoring the HTTP client after each test.

    Returns:
        Module, session mock, mutable response and session factory.
    """
    path = Path(__file__).resolve().parents[1] / "tavily_search.py"
    spec = importlib.util.spec_from_file_location("tavily_references_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    response = SimpleNamespace(status=200, content_length=None, chunks=[])

    async def chunks(size):
        for chunk in response.chunks:
            yield chunk

    response.content = SimpleNamespace(iter_chunked=chunks)
    post_context = AsyncMock()
    post_context.__aenter__.return_value = response
    session = Mock()
    session.post.return_value = post_context
    session_context = AsyncMock()
    session_context.__aenter__.return_value = session
    factory = Mock(return_value=session_context)
    monkeypatch.setattr(module.aiohttp, "ClientSession", factory)
    return SimpleNamespace(
        module=module, session=session, response=response, factory=factory
    )


CONFIG = {"provider_settings": {"websearch_tavily_key": ["test-key", "unused-key"]}}


@pytest.mark.asyncio
async def test_source_linked_images_preserve_provenance_and_request_limits(tavily_env):
    tavily_env.response.chunks = [
        json.dumps(
            {
                "images": [{"url": "https://unrelated.example/other.png"}],
                "results": [
                    {
                        "url": "https://official.example/character",
                        "title": "Sherry",
                        "content": "Official character profile.",
                        "images": [
                            {
                                "url": "https://cdn.example/sherry.png",
                                "description": "Sherry character art.",
                            }
                        ],
                    }
                ],
            }
        ).encode()
    ]
    result = await tavily_env.module.search_tavily_references(
        " Sherry character art ", CONFIG
    )
    assert result == [
        {
            "image_url": "https://cdn.example/sherry.png",
            "source_website_url": "https://official.example/character",
            "caption": "Image description: Sherry character art.\nSource title: Sherry\nSource excerpt: Official character profile.",
        }
    ]
    args, kwargs = tavily_env.session.post.call_args
    assert args == ("https://api.tavily.com/search",)
    assert kwargs["headers"] == {"Authorization": "Bearer test-key"}
    assert kwargs["allow_redirects"] is False
    assert kwargs["json"] == {
        "query": "Sherry character art",
        "search_depth": "basic",
        "max_results": 3,
        "include_images": True,
        "include_image_descriptions": True,
        "include_answer": False,
        "include_raw_content": False,
        "auto_parameters": False,
    }
    assert tavily_env.factory.call_args.kwargs["timeout"].total == 30
    assert tavily_env.factory.call_args.kwargs["trust_env"] is False
    assert CONFIG["provider_settings"]["websearch_tavily_key"] == [
        "test-key",
        "unused-key",
    ]


@pytest.mark.asyncio
async def test_deduplicates_limits_and_skips_unusable_metadata(tavily_env):
    tavily_env.response.chunks = [
        json.dumps(
            {
                "results": [
                    {
                        "url": "https://official.example/character",
                        "title": "x" * 600,
                        "content": "y" * 3000,
                        "images": [
                            None,
                            "https://untyped.example/a.png",
                            {"url": "https://credentials:secret@example.com/a.png"},
                            {"url": "file:///private.png"},
                            {"url": "https://[invalid/a.png"},
                            {
                                "url": "https://cdn.example/a.png",
                                "description": "z" * 2000,
                            },
                            {"url": "https://cdn.example/a.png"},
                            {"url": "https://cdn.example/b.png"},
                            {"url": "https://cdn.example/c.png"},
                            {"url": "https://cdn.example/d.png"},
                        ],
                    }
                ]
            }
        ).encode()
    ]
    references = await tavily_env.module.search_tavily_references("character", CONFIG)
    assert [item["image_url"] for item in references] == [
        "https://cdn.example/a.png",
        "https://cdn.example/b.png",
        "https://cdn.example/c.png",
    ]
    assert "x" * 301 not in references[0]["caption"]
    assert "y" * 1501 not in references[0]["caption"]
    assert "z" * 1001 not in references[0]["caption"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        {},
        {"provider_settings": None},
        {"provider_settings": {"websearch_tavily_key": "not-a-list"}},
        {"provider_settings": {"websearch_tavily_key": [None, " "]}},
    ],
)
async def test_missing_keys_fail_before_network(tavily_env, config):
    with pytest.raises(ValueError, match="Configure a Tavily API key"):
        await tavily_env.module.search_tavily_references("character", config)
    tavily_env.factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("query", [None, "", "  ", "x" * 1501])
async def test_invalid_query_fails_before_network(tavily_env, query):
    with pytest.raises(ValueError, match="queries"):
        await tavily_env.module.search_tavily_references(query, CONFIG)
    tavily_env.factory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 401, 429, 500])
async def test_http_errors_have_no_retries_or_body_exposure(tavily_env, status):
    tavily_env.response.status = status
    tavily_env.response.chunks = [b'"credential-reflection-test-key"']
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as error:
        await tavily_env.module.search_tavily_references("character", CONFIG)
    assert "credential-reflection" not in str(error.value)
    assert "test-key" not in str(error.value)
    tavily_env.session.post.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"[]",
        b'{"results": {}}',
        b'{"images": [{"url": "https://cdn.example/a.png"}]}',
        b'{"results": [{"url": "javascript:alert(1)", "images": [{"url": "https://cdn.example/a.png"}]}]}',
    ],
)
async def test_rejects_invalid_or_unsourced_results(tavily_env, body):
    tavily_env.response.chunks = [body]
    with pytest.raises(RuntimeError):
        await tavily_env.module.search_tavily_references("character", CONFIG)


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", [None, 2 * 1024 * 1024 + 1])
async def test_limits_declared_and_streamed_response_size(tavily_env, declared):
    tavily_env.response.content_length = declared
    tavily_env.response.chunks = [b"x" * (2 * 1024 * 1024), b"x"]
    with pytest.raises(RuntimeError, match="exceeds 2 MiB"):
        await tavily_env.module.search_tavily_references("character", CONFIG)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["source", "image"])
@pytest.mark.parametrize(
    "url",
    [
        "",
        "https://example.com/a\nInjected content",
        "https://example.com/a\t.png",
        "https://example.com/a\x7f.png",
        "https://example.com/a b.png",
        "https://example.com:invalid/a.png",
        "https://example.com:99999/a.png",
        "https://example.com/" + "x" * 8192,
    ],
)
async def test_rejects_invalid_source_and_image_urls(tavily_env, field, url):
    tavily_env.response.chunks = [
        json.dumps(
            {
                "results": [
                    {
                        "url": url
                        if field == "source"
                        else "https://official.example/character",
                        "images": [
                            {
                                "url": url
                                if field == "image"
                                else "https://cdn.example/reference.png"
                            }
                        ],
                    }
                ]
            }
        ).encode()
    ]
    with pytest.raises(RuntimeError, match="no images with an associated source"):
        await tavily_env.module.search_tavily_references("character", CONFIG)
