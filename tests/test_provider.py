"""Exercise the image provider using real SDK models and offline transports."""

import asyncio
import base64
import copy
import importlib.util
import json
import sys
from email import policy
from email.parser import BytesParser
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import pytest_asyncio
from astrbot.core.message.components import Image, Plain
from openai import AsyncOpenAI, InternalServerError
from openai.types.responses import Response

_package = ModuleType("responses_image_provider_tests")
_package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[_package.__name__] = _package
_spec = importlib.util.spec_from_file_location(
    f"{_package.__name__}.provider",
    Path(__file__).resolve().parents[1] / "provider.py",
)
assert _spec and _spec.loader
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
ResponsesImageClient = _module.ResponsesImageClient
ReferenceSearchError = _module.ReferenceSearchError

REFERENCE = {
    "image_url": "https://images.example/character.png",
    "source_website_url": "https://official.example/character",
    "caption": "Official character illustration",
}

PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1Pe"
    "AAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def make_response(output: list[dict], **overrides) -> Response:
    """Validate a fixture against the installed OpenAI SDK response schema.

    Args:
        output: Response output items.
        **overrides: Additional or replacement response fields.

    Returns:
        A real SDK response model.
    """
    data = {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "test-model",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 0},
            "output_tokens": 4,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 14,
        },
        **overrides,
    }
    return Response.model_validate(data)


def image_item(image_id: str = "ig_test", **overrides) -> dict:
    """Build a completed native image output item.

    Args:
        image_id: Image output identifier.
        **overrides: Image fields to replace.

    Returns:
        An image item with a valid one-pixel PNG payload.
    """
    return {
        "id": image_id,
        "type": "image_generation_call",
        "status": "completed",
        "result": PNG_BASE64,
        **overrides,
    }


def text_item(text: str, message_id: str = "msg_test") -> dict:
    """Build a completed assistant text output item.

    Args:
        text: Assistant message text.
        message_id: Output message identifier.

    Returns:
        An assistant output message.
    """
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


@pytest_asyncio.fixture
async def make_provider():
    """Create real providers and close their unused HTTP clients afterwards.

    Yields:
        A factory accepting provider configuration overrides.
    """
    providers = []

    def factory(provider_class=_module.ProviderOpenAIResponses, **overrides):
        """Create a provider with offline credentials.

        Args:
            provider_class: The adapter class to instantiate with offline settings.
            **overrides: Provider configuration overrides.

        Returns:
            A provider instance.
        """
        provider = provider_class(
            {
                "id": "test-responses-images",
                "type": "openai_responses",
                "provider": "openai",
                "model": "test-model",
                "key": ["offline-test-key"],
                "api_base": "https://example.invalid/v1",
                **overrides,
            },
            {},
        )
        providers.append(provider)
        return provider

    yield factory
    for provider in providers:
        await provider.client.close()


@pytest.mark.asyncio
async def test_generic_responses_provider_supports_all_isolated_image_operations(
    make_provider, monkeypatch, tmp_path
):
    provider = make_provider(
        provider_class=_module.ProviderOpenAIResponses,
        custom_extra_body={
            "input": "DO_NOT_REPLAY",
            "previous_response_id": "old-response",
            "tools": [{"type": "image_generation", "quality": "high"}],
        },
    )
    assert type(provider) is _module.ProviderOpenAIResponses
    original_config = copy.deepcopy(provider.provider_config)
    requests = []

    async def handle(request):
        """Return native search, vision or image output through a real SDK client.

        Args:
            request: HTTP request captured by the offline transport.

        Returns:
            A complete Responses API fixture for the requested operation.
        """
        body = json.loads(request.content)
        requests.append(body)
        kind = body["tools"][0]["type"] if body["tools"] else "vision"
        if kind == "web_search":
            output = [search_item([{"type": "image_result", **REFERENCE}])]
        elif kind == "vision":
            output = [text_item('{"matched": true, "appearance": "Red ribbon."}')]
        else:
            output = [image_item()]
        return httpx.Response(200, json=make_response(output).model_dump(mode="json"))

    await provider.client.close()
    provider.client = AsyncOpenAI(
        api_key="offline",
        base_url="https://example.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    shared_client = provider.client
    ordinary_parser = Mock(
        side_effect=AssertionError("The ordinary parser would lose image output")
    )
    monkeypatch.setattr(provider, "_parse_response", ordinary_parser)
    worker = _module.ResponsesImageClient(provider)
    refs = await worker.search_reference_images("Sherry", "Draw Sherry")
    assert refs == [REFERENCE]
    path = tmp_path / "reference.png"
    path.write_bytes(base64.b64decode(PNG_BASE64))
    assert await worker.verify_reference_image(
        "Sherry", "Draw Sherry", refs[0], str(path), collect_details=True
    ) == {"appearance": "Red ribbon."}
    result = await worker.generate_image(
        "Change only the background",
        [str(path)],
        operation="edit",
        image_model="selected-image",
    )
    assert result.result_chain.chain[0].file == "base64://" + PNG_BASE64
    assert result.usage.total == 14
    assert len(requests) == 3
    assert requests[2]["model"] == provider.get_model()
    assert requests[2]["tools"] == [
        {
            "type": "image_generation",
            "quality": "high",
            "action": "edit",
            "model": "selected-image",
        }
    ]
    assert len(requests[2]["input"]) == 1
    assert len(requests[2]["input"][0]["content"]) == 2
    assert all(
        "previous_response_id" not in request and request["store"] is False
        for request in requests
    )
    assert "DO_NOT_REPLAY" not in json.dumps(requests)
    assert provider.provider_config == original_config
    assert provider.client is shared_client and not shared_client.is_closed()
    ordinary_parser.assert_not_called()


def test_image_client_rejects_non_responses_adapters():
    with pytest.raises(TypeError, match="Responses provider"):
        _module.ResponsesImageClient(object())


@pytest.mark.asyncio
async def test_image_only_is_usable_output(make_provider):
    provider = make_provider()
    response = make_response([image_item()])
    result = await ResponsesImageClient(provider)._parse_image_response(response)
    assert result.role == "assistant"
    assert result.completion_text == ""
    assert len(result.result_chain.chain) == 1
    assert isinstance(result.result_chain.chain[0], Image)
    assert result.result_chain.chain[0].file == f"base64://{PNG_BASE64}"
    assert result.raw_completion is response
    assert result.usage.input_other == 7
    assert result.usage.input_cached == 3
    assert result.usage.output == 4


@pytest.mark.asyncio
async def test_interleaved_text_and_multiple_images_keep_source_order(make_provider):
    response = make_response(
        [
            text_item("Before"),
            image_item("ig_1"),
            text_item("After"),
            image_item("ig_2"),
        ]
    )
    result = await ResponsesImageClient(make_provider())._parse_image_response(response)
    assert [type(part) for part in result.result_chain.chain] == [
        Plain,
        Image,
        Plain,
        Image,
    ]
    assert result.result_chain.chain[0].text == "Before"
    assert result.result_chain.chain[2].text == "After"


@pytest.mark.asyncio
async def test_worker_ignores_unrequested_reasoning_and_function_calls(make_provider):
    response = make_response(
        [
            {
                "id": "rs_test",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "I should draw."}],
                "encrypted_content": "opaque-reasoning-state",
            },
            image_item(),
            {
                "id": "fc_test",
                "type": "function_call",
                "call_id": "call_weather",
                "name": "weather",
                "arguments": '{"city":"Tokyo"}',
                "status": "completed",
            },
        ]
    )
    result = await ResponsesImageClient(make_provider())._parse_image_response(response)
    assert result.role == "assistant"
    assert result.tools_call_args == []
    assert result.tools_call_name == []
    assert result.tools_call_ids == []
    assert not result.reasoning_content
    assert not result.reasoning_signature
    assert isinstance(result.result_chain.chain[0], Image)
    assert result.raw_completion is response
    assert result.usage.input_cached == 3


@pytest.mark.asyncio
async def test_image_base64_whitespace_is_normalized(make_provider):
    encoded = " \n" + PNG_BASE64[:30] + "\r\n" + PNG_BASE64[30:] + "\t"
    result = await ResponsesImageClient(make_provider())._parse_image_response(
        make_response([image_item(result=encoded)])
    )
    assert result.result_chain.chain[0].file == f"base64://{PNG_BASE64}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"status": "failed", "result": None}, "Image generation failed"),
        ({"result": None}, "no image data"),
        ({"result": ""}, "no image data"),
        ({"result": "not@@base64"}, "invalid Base64"),
    ],
)
async def test_unusable_image_only_returns_explanatory_error(
    make_provider, overrides, message
):
    with pytest.raises(RuntimeError, match=message):
        await ResponsesImageClient(make_provider())._parse_image_response(
            make_response([image_item(**overrides)])
        )


@pytest.mark.asyncio
async def test_unfinished_image_is_not_sent(make_provider):
    with pytest.raises(RuntimeError, match="without an image"):
        await ResponsesImageClient(make_provider())._parse_image_response(
            make_response([image_item(status="generating"), text_item("Still working")])
        )


@pytest.mark.asyncio
async def test_bad_image_does_not_discard_valid_text_and_other_images(make_provider):
    result = await ResponsesImageClient(make_provider())._parse_image_response(
        make_response(
            [image_item(status="failed", result=None), text_item("Done"), image_item()]
        ),
    )
    assert [type(part) for part in result.result_chain.chain] == [Plain, Image]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "status": "failed",
                "error": {"code": "server_error", "message": "Upstream failed"},
            },
            "Upstream failed",
        ),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
            },
            "content filter",
        ),
    ],
)
async def test_top_level_errors_are_not_hidden_by_image_data(
    make_provider, overrides, message
):
    with pytest.raises(RuntimeError, match=message):
        await ResponsesImageClient(make_provider())._parse_image_response(
            make_response([image_item()], **overrides)
        )


@pytest.mark.asyncio
async def test_empty_response_still_raises(make_provider):
    with pytest.raises(RuntimeError, match="without an image"):
        await ResponsesImageClient(make_provider())._parse_image_response(
            make_response([])
        )


@pytest.fixture
def images_api_transport():
    """Install offline SDK transports for Images API boundary tests.

    Returns:
        An async factory returning captured HTTP requests.
    """

    async def install(provider, payload=None, status=200):
        """Replace a provider's transport with a deterministic response.

        Args:
            provider: Real provider owning the SDK client.
            payload: Optional response body.
            status: HTTP status code returned for every request.

        Returns:
            A mutable list of captured HTTP requests.
        """
        requests = []
        if payload is None:
            payload = {
                "created": 0,
                "data": [{"b64_json": PNG_BASE64}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "total_tokens": 14,
                    "input_tokens_details": {"text_tokens": 8, "image_tokens": 2},
                },
            }

        async def handle(request):
            """Capture the complete multipart or JSON request before responding.

            Args:
                request: SDK request intercepted before any network traffic.

            Returns:
                The configured HTTP response.
            """
            await request.aread()
            requests.append(request)
            return httpx.Response(status, json=payload)

        await provider.client.close()
        provider.client = AsyncOpenAI(
            api_key="offline",
            base_url="https://example.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        )
        return requests

    return install


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_class", [_module.ProviderOpenAIResponses, _module.ProviderOpenAIOfficial]
)
@pytest.mark.parametrize(
    "operation, reference_count", [("new", 0), ("new", 1), ("edit", 1), ("edit", 2)]
)
async def test_images_api_routes_real_sdk_requests_and_returns_pixels(
    make_provider,
    images_api_transport,
    tmp_path,
    provider_class,
    operation,
    reference_count,
):
    config = {
        "input": "DO_NOT_REPLAY",
        "previous_response_id": "DO_NOT_REPLAY",
        "instructions": "DO_NOT_REPLAY",
        "store": True,
        "stream": True,
        "n": 99,
        "quality": "high",
        "reasoning": {"effort": "high"},
        "tools": [
            {"type": "web_search"},
            {
                "type": "image_generation",
                "model": "tool-image",
                "quality": "low",
                "output_format": "png",
            },
        ],
    }
    provider = make_provider(
        provider_class=provider_class, custom_extra_body=copy.deepcopy(config)
    )
    requests = await images_api_transport(provider)
    shared_client = provider.client
    path = tmp_path / "PRIVATE_LOCAL_PATH.png"
    path.write_bytes(base64.b64decode(PNG_BASE64))
    result = await _module.ImagesAPIClient(provider).generate_image(
        "Only this task",
        [str(path)] * reference_count,
        operation=operation,
        image_model=" selected-image ",
    )
    assert len(requests) == 1
    request = requests[0]
    assert request.url.path == (
        "/v1/images/edits" if reference_count else "/v1/images/generations"
    )
    if reference_count:
        message = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: "
            + request.headers["content-type"].encode()
            + b"\r\n\r\n"
            + request.content
        )
        parts = list(message.iter_parts())
        fields = {
            part.get_param("name", header="content-disposition"): part.get_payload(
                decode=True
            ).decode()
            for part in parts
            if not part.get_filename()
        }
        files = [part for part in parts if part.get_filename()]
        assert len(files) == reference_count
        assert all(part.get_payload(decode=True) == path.read_bytes() for part in files)
        assert all(part.get_content_type() == "image/png" for part in files)
        assert {
            part.get_param("name", header="content-disposition") for part in files
        } == {"image" if reference_count == 1 else "image[]"}
    else:
        fields = json.loads(request.content)
    assert fields == {
        "model": "selected-image",
        "prompt": "Only this task",
        "n": "1" if reference_count else 1,
        "quality": "high",
        "output_format": "png",
    }
    assert b"DO_NOT_REPLAY" not in request.content
    assert b"PRIVATE_LOCAL_PATH" not in request.content
    assert result.result_chain.chain[0].file == "base64://" + PNG_BASE64
    assert result.usage.input_other == 10 and result.usage.output == 4
    assert provider.provider_config["custom_extra_body"] == config
    assert provider.client is shared_client and not shared_client.is_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "panel, tool_model, expected",
    [
        ("panel-image", "tool-image", "panel-image"),
        ("", "tool-image", "tool-image"),
        ("", None, "worker-image"),
    ],
)
async def test_images_api_model_fallbacks(
    make_provider, images_api_transport, panel, tool_model, expected
):
    provider = make_provider(
        model="worker-image",
        custom_extra_body={
            "tools": [{"type": "image_generation", "model": tool_model}]
        },
    )
    requests = await images_api_transport(provider)
    await _module.ImagesAPIClient(provider).generate_image(
        "Draw", [], image_model=panel
    )
    assert json.loads(requests[0].content) == {
        "model": expected,
        "prompt": "Draw",
        "n": 1,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 429, 500])
async def test_images_api_does_not_retry_or_fall_back(
    make_provider, images_api_transport, status
):
    from openai import APIStatusError

    provider = make_provider()
    requests = await images_api_transport(
        provider, {"error": {"message": "Offline failure"}}, status
    )
    with pytest.raises(APIStatusError):
        await _module.ImagesAPIClient(provider).generate_image("Draw", [])
    assert len(requests) == 1 and requests[0].url.path == "/v1/images/generations"
    assert not provider.client.is_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "items",
    [
        [],
        [{}],
        [{"b64_json": "not_base64"}],
        [{"b64_json": "bm90IGFuIGltYWdl"}],
        [{"b64_json": PNG_BASE64}] * 5,
    ],
)
async def test_images_api_rejects_empty_or_invalid_output(
    make_provider, images_api_transport, items
):
    provider = make_provider()
    requests = await images_api_transport(provider, {"created": 0, "data": items})
    with pytest.raises((RuntimeError, ValueError, OSError)):
        await _module.ImagesAPIClient(provider).generate_image("Draw", [])
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_images_api_downloads_url_pixels_with_existing_guard(
    make_provider, images_api_transport, monkeypatch
):
    provider = make_provider()
    await images_api_transport(
        provider, {"created": 0, "data": [{"url": "https://images.example/result.png"}]}
    )
    download = AsyncMock(
        return_value=(
            base64.b64decode(PNG_BASE64),
            "https://images.example/result.png",
            "image/png",
        )
    )
    monkeypatch.setattr(_module, "_fetch_reference_bytes", download)
    result = await _module.ImagesAPIClient(provider).generate_image("Draw", [])
    download.assert_awaited_once_with(
        "https://images.example/result.png",
        _module.MAX_IMAGE_BYTES,
        "image/png,image/jpeg,image/webp",
    )
    assert result.result_chain.chain[0].file == "base64://" + PNG_BASE64


@pytest.mark.asyncio
async def test_images_api_rejects_private_output_urls(
    make_provider, images_api_transport
):
    provider = make_provider()
    requests = await images_api_transport(
        provider, {"created": 0, "data": [{"url": "http://127.0.0.1/private.png"}]}
    )
    with pytest.raises(ValueError):
        await _module.ImagesAPIClient(provider).generate_image("Draw", [])
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "edit_without_image",
        "missing_file",
        "invalid_file",
        "unsupported_format",
        "too_many_files",
        "wrong_operation",
    ],
)
async def test_images_api_invalid_inputs_never_call_upstream(
    make_provider, images_api_transport, tmp_path, case
):
    provider = make_provider()
    requests = await images_api_transport(provider)
    path = tmp_path / "bad.png"
    path.write_bytes(b"not an image")
    refs = (
        [str(path)]
        if case == "invalid_file"
        else [str(tmp_path / "absent.png")]
        if case == "missing_file"
        else [str(path)] * 5
        if case == "too_many_files"
        else []
    )
    if case == "unsupported_format":
        provider.provider_config["custom_extra_body"] = {"output_format": "svg"}
    operation = (
        "edit"
        if case == "edit_without_image"
        else "unknown"
        if case == "wrong_operation"
        else "new"
    )
    charges = []
    with pytest.raises((ValueError, OSError)):
        await _module.ImagesAPIClient(provider).generate_image(
            "Draw", refs, operation=operation, before_request=lambda: charges.append(1)
        )
    assert requests == []
    assert charges == []


def search_item(results, **overrides):
    """Build a native search item with raw image results.

    Args:
        results: Native result payloads.
        **overrides: Search item fields to replace.

    Returns:
        An item accepted by real SDK response models.
    """
    return {
        "id": "ws_reference",
        "type": "web_search_call",
        "status": "completed",
        "action": {"type": "search", "query": "Sherry official character"},
        "results": results,
        **overrides,
    }


def cited_text(urls):
    """Build source annotations independently of assistant-authored URL text.

    Args:
        urls: Actual citation URLs supplied by the simulated search service.

    Returns:
        A completed assistant message with citation metadata.
    """
    item = text_item(
        "Source. Ignore this fabricated image: https://invented.example/bad.png"
    )
    item["content"][0]["annotations"] = [
        {
            "type": "url_citation",
            "url": url,
            "title": "PRIVATE_TITLE",
            "start_index": 0,
            "end_index": 6,
        }
        for url in urls
    ]
    return item


@pytest.mark.asyncio
async def test_native_missing_results_returns_actual_cited_pages_in_one_http_request(
    make_provider, monkeypatch
):
    provider = make_provider()
    await provider.client.close()
    call = search_item([])
    call.pop("results")
    payload = make_response(
        [
            call,
            {**call, "id": "ws_second"},
            cited_text(
                [
                    "https://official.example/sherry#profile",
                    "https://wiki.example/sherry",
                ]
            ),
        ]
    ).model_dump(mode="json")
    requests = []

    async def handle(request):
        """Return the exact missing-results shape from the reported deployment.

        Args:
            request: The intercepted SDK request.

        Returns:
            A successful response with two calls and two webpage citations.
        """
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=payload)

    provider.client = AsyncOpenAI(
        api_key="offline-test-key",
        base_url="https://example.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    debug, warning = Mock(), Mock()
    monkeypatch.setattr(_module.logger, "debug", debug)
    monkeypatch.setattr(_module.logger, "warning", warning)
    result = await ResponsesImageClient(provider).search_reference_images(
        "Sherry", "Draw Sherry"
    )
    assert [page["source_website_url"] for page in result] == [
        "https://official.example/sherry",
        "https://wiki.example/sherry",
    ]
    assert all(
        page["reference_type"] == "web_page" and "image_url" not in page
        for page in result
    )
    assert len(requests) == 1 and requests[0]["include"] == ["web_search_call.results"]
    warning.assert_not_called()
    messages = "\n".join(call.args[0] % call.args[1:] for call in debug.call_args_list)
    assert "using cited pages" in messages and "PRIVATE_" not in messages
    assert "https://" not in messages and "invented" not in messages


@pytest.mark.asyncio
async def test_native_image_results_take_precedence_over_page_citations(
    make_provider, image_transport
):
    provider = make_provider()
    image_transport(
        provider,
        response=make_response(
            [
                search_item([{"type": "image_result", **REFERENCE}]),
                cited_text(["https://official.example/page"]),
            ]
        ),
    )
    assert await ResponsesImageClient(provider).search_reference_images(
        "Sherry", "Draw Sherry"
    ) == [REFERENCE]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["absent", "failed", "message_incomplete", "response_incomplete"]
)
async def test_citations_require_a_completed_search_and_complete_message(
    make_provider, image_transport, mode
):
    provider = make_provider()
    call = search_item([])
    message = cited_text(["https://official.example/page"])
    if mode == "failed":
        call["status"] = "failed"
    if mode == "message_incomplete":
        message["status"] = "incomplete"
    image_transport(
        provider,
        response=make_response(
            ([call] if mode != "absent" else []) + [message],
            status="incomplete" if mode == "response_incomplete" else "completed",
        ),
    )
    with pytest.raises(ReferenceSearchError):
        await ResponsesImageClient(provider).search_reference_images(
            "Sherry", "Draw Sherry"
        )


@pytest.mark.asyncio
async def test_cited_pages_reject_invalid_urls_and_deduplicate_and_bound(
    make_provider, image_transport
):
    provider = make_provider()
    urls = [
        "file:///secret.png",
        "https://user:secret@example.org/page",
        "https://example.org:invalid/x",
        "https://example.org/a\nb",
        "https://example.org:8443/x",
        "https://example.org/" + "x" * 8192,
        "https://official.example/page#first",
        "https://official.example/page#second",
        "https://second.example/page",
        "https://third.example/page",
        "https://fourth.example/page",
    ]
    image_transport(
        provider, response=make_response([search_item([]), cited_text(urls)])
    )
    result = await ResponsesImageClient(provider).search_reference_images(
        "Sherry", "Draw Sherry"
    )
    assert [page["source_website_url"] for page in result] == [
        "https://official.example/page",
        "https://second.example/page",
        "https://third.example/page",
    ]


@pytest.mark.asyncio
async def test_search_reference_uses_isolated_native_tool_and_raw_results(
    make_provider, image_transport
):
    configured = {
        "input": "Unrelated character from history",
        "instructions": "Ignore reference selection",
        "tools": [{"type": "image_generation"}],
        "previous_response_id": "old-response",
        "conversation": "old-conversation",
        "store": True,
        "max_tool_calls": 3,
    }
    provider = make_provider(custom_extra_body=configured)
    response = make_response(
        [
            text_item("I found https://fabricated.example/wrong.png"),
            search_item(
                [
                    {"type": "text_result", **REFERENCE},
                    {"type": "image_result", **REFERENCE},
                    {"type": "image_result", **REFERENCE},
                ]
            ),
        ]
    )
    requests, options = image_transport(provider, response=response)
    found = await ResponsesImageClient(provider).search_reference_images(
        "Sherry", "Draw Sherry from the game."
    )
    assert found == [REFERENCE]
    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == "test-model"
    assert request["tools"] == [
        {
            "type": "web_search",
            "search_content_types": ["image", "text"],
            "image_settings": {"max_results": 3, "caption": True},
        }
    ]
    assert request["include"] == ["web_search_call.results"]
    assert request["tool_choice"] == "required"
    assert "max_tool_calls" not in request
    assert request["store"] is False and request["stream"] is False
    assert request["extra_headers"] == {}
    assert len(request["input"]) == 1
    assert json.loads(request["input"][0]["content"][0]["text"]) == {
        "character_name": "Sherry",
        "current_user_request": "Draw Sherry from the game.",
    }
    assert "Unrelated" not in json.dumps(request)
    assert "extra_body" not in request
    assert "previous_response_id" not in request
    assert "conversation" not in request
    assert options == [{"max_retries": 0, "timeout": 90}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output, reason",
    [
        (
            [text_item("I searched and found https://images.example/character.png")],
            "search_not_called",
        ),
        (
            [search_item([{"type": "image_result", **REFERENCE}], status="failed")],
            "incomplete_response",
        ),
        ([search_item({"image_url": REFERENCE["image_url"]})], "missing_results"),
        (
            [search_item([None, "not a result", {"type": "text_result", **REFERENCE}])],
            "no_image_results",
        ),
        (
            [
                search_item(
                    [{"type": "image_result", "image_url": REFERENCE["image_url"]}]
                )
            ],
            "no_usable_reference",
        ),
    ],
)
async def test_search_never_uses_assistant_urls_or_unusable_native_results(
    make_provider, image_transport, output, reason
):
    provider = make_provider()
    image_transport(provider, response=make_response(output))
    with pytest.raises(ReferenceSearchError, match=f"^{reason}$"):
        await ResponsesImageClient(provider).search_reference_images(
            "Sherry", "Draw Sherry"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "file:///private/image.png",
        "https://name:secret@example.org/image.png",
        "https://example.org/with space.png",
        "https://example.org/bad\nimage.png",
        "https://example.org:invalid/image.png",
        "https:///missing-host.png",
        "https://[invalid/image.png",
        "https://example.org/" + "a" * 8192,
    ],
)
@pytest.mark.parametrize("key", ["image_url", "source_website_url"])
async def test_search_rejects_unclean_image_and_citation_urls(
    make_provider, image_transport, url, key
):
    provider = make_provider()
    response = make_response(
        [search_item([{"type": "image_result", **REFERENCE, key: url}])]
    )
    image_transport(provider, response=response)
    with pytest.raises(ReferenceSearchError, match="^no_usable_reference$"):
        await ResponsesImageClient(provider).search_reference_images(
            "Sherry", "Draw Sherry"
        )


@pytest.mark.asyncio
async def test_search_bounds_results_and_captions(make_provider, image_transport):
    provider = make_provider()
    response = make_response(
        [
            search_item(
                [
                    {
                        "type": "image_result",
                        **REFERENCE,
                        "image_url": f"https://images.example/{index}.png",
                        "caption": "x" * 3000,
                    }
                    for index in range(8)
                ]
            )
        ]
    )
    image_transport(provider, response=response)
    result = await ResponsesImageClient(provider).search_reference_images(
        "Sherry", "Draw Sherry"
    )
    assert len(result) == 3
    assert all(len(item["caption"]) == 2000 for item in result)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["incomplete", "failed", "in_progress"])
async def test_search_rejects_incomplete_response(
    make_provider, image_transport, status
):
    provider = make_provider()
    image_transport(
        provider,
        response=make_response(
            [search_item([{"type": "image_result", **REFERENCE}])], status=status
        ),
    )
    with pytest.raises(ReferenceSearchError, match="^incomplete_response$"):
        await ResponsesImageClient(provider).search_reference_images(
            "Sherry", "Draw Sherry"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("reject_call_limit", [False, True])
async def test_search_sdk_serializes_new_native_fields_and_retains_raw_results(
    make_provider,
    reject_call_limit,
):
    provider = make_provider()
    await provider.client.close()
    requests = []

    async def handle(request):
        requests.append(json.loads(request.content))
        if reject_call_limit and "max_tool_calls" in requests[-1]:
            return httpx.Response(
                400, json={"detail": "Unsupported parameter: max_tool_calls"}
            )
        payload = make_response(
            [search_item([{"type": "image_result", **REFERENCE}])]
        ).model_dump(mode="json")
        return httpx.Response(200, json=payload)

    provider.client = AsyncOpenAI(
        api_key="offline-test-key",
        base_url="https://example.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    result = await ResponsesImageClient(provider).search_reference_images(
        "Sherry", "Draw Sherry"
    )
    assert result == [REFERENCE]
    assert len(requests) == 1
    assert "max_tool_calls" not in requests[0]
    assert [tool["type"] for tool in requests[0]["tools"]] == ["web_search"]
    assert requests[0]["tools"][0]["search_content_types"] == ["image", "text"]
    assert requests[0]["tools"][0]["image_settings"] == {
        "max_results": 3,
        "caption": True,
    }
    assert requests[0]["include"] == ["web_search_call.results"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_kind, reason",
    [
        ("missing", "missing_results"),
        ("null", "missing_results"),
        ("malformed", "missing_results"),
        ("empty", "no_image_results"),
        ("text_only", "no_image_results"),
        ("invalid_image", "no_usable_reference"),
        ("not_called", "search_not_called"),
        ("image_instead", "search_not_called"),
        ("incomplete", "incomplete_response"),
    ],
)
async def test_search_diagnoses_http_200_without_exposing_response_content(
    make_provider, monkeypatch, response_kind, reason
):
    """Distinguish successful transport from usable search output.

    Args:
        make_provider: Fixture creating a real provider instance.
        monkeypatch: Fixture restoring the diagnostic logger.
        response_kind: A specific malformed or incomplete upstream result.
        reason: Expected safe error code for user-facing diagnostics.
    """
    provider = make_provider()
    await provider.client.close()
    item = search_item([])
    item["action"]["query"] = "PRIVATE_QUERY"
    if response_kind == "missing":
        item.pop("results")
    elif response_kind == "null":
        item["results"] = None
    elif response_kind == "malformed":
        item["results"] = {"private": "PRIVATE_RESPONSE"}
    elif response_kind == "text_only":
        item["results"] = [{"type": "text_result", "text": "PRIVATE_RESPONSE"}]
    elif response_kind == "invalid_image":
        item["results"] = [
            {
                "type": "image_result",
                "image_url": "https://private.example/PRIVATE_IMAGE",
                "caption": "PRIVATE_CAPTION",
            }
        ]
    output = [text_item("PRIVATE_ASSISTANT"), item]
    if response_kind == "not_called":
        output = output[:1]
    elif response_kind == "image_instead":
        output = [image_item()]
    payload = make_response(
        output, status="incomplete" if response_kind == "incomplete" else "completed"
    ).model_dump(mode="json")
    requests = []

    async def handle(request):
        """Return an HTTP success with deliberately unusable tool results.

        Args:
            request: The SDK request intercepted by the offline transport.

        Returns:
            A simulated successful JSON response.
        """
        requests.append(request)
        return httpx.Response(200, json=payload)

    warning = Mock()
    debug = Mock()
    monkeypatch.setattr(_module.logger, "warning", warning)
    monkeypatch.setattr(_module.logger, "debug", debug)
    provider.client = AsyncOpenAI(
        api_key="offline-test-key",
        base_url="https://example.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    with pytest.raises(ReferenceSearchError, match=f"^{reason}$") as raised:
        await ResponsesImageClient(provider).search_reference_images(
            "PRIVATE_CHARACTER", "PRIVATE_REQUEST"
        )
    assert str(raised.value) == reason
    assert len(requests) == 1
    warning.assert_called_once()
    diagnostic = "\n".join(
        call.args[0] % call.args[1:] for call in warning.call_args_list
    )
    assert reason in diagnostic
    assert "PRIVATE_" not in diagnostic
    assert "https://" not in diagnostic
    assert PNG_BASE64 not in diagnostic
    if response_kind not in {"incomplete", "not_called", "image_instead"}:
        assert "search_calls=1" in diagnostic
        assert "completed_search_calls=1" in diagnostic
    debug.assert_called_once()
    assert debug.call_args.args[0] == "Native reference search diagnostic: %s"
    encoded = debug.call_args.args[1]
    assert "PRIVATE_" not in encoded and "https://" not in encoded
    assert PNG_BASE64 not in encoded
    assert not hasattr(raised.value, "diagnostic")
    structure = json.loads(encoded)
    assert structure["http_status"] == 200
    assert structure["sent"] == {"include_results": True, "tools": {"web_search": 1}}
    assert structure["raw_sdk_match"] is True
    assert "sdk" not in structure
    assert len(encoded) <= 2500
    if response_kind in {"missing", "null", "malformed", "empty"}:
        assert (
            structure["wire"]["search_calls"][0]["results_type"]
            == {
                "missing": "missing",
                "null": "null",
                "malformed": "object",
                "empty": "list",
            }[response_kind]
        )


def test_search_diagnostic_records_only_whitelisted_structure():
    """Keep untrusted keys, enum values and payloads out of shareable diagnostics."""
    parsed = make_response([search_item([])])
    wire = parsed.model_dump(mode="json")
    call = wire["output"][0]
    call["PRIVATE_KEY"] = "PRIVATE_VALUE"
    call["results"] = {"PRIVATE_KEY": "PRIVATE_RESULT"}
    call["action"] = {
        "type": "PRIVATE_ACTION",
        "query": "PRIVATE_QUERY",
        "sources": [{"type": "url", "url": "https://private.example/PRIVATE_SOURCE"}],
        "results": [{"type": "image_result", "image_url": "PRIVATE_IMAGE"}],
        "PRIVATE_ACTION_KEY": "PRIVATE_ACTION_VALUE",
    }
    wire["output"].extend(
        [
            {"type": "PRIVATE_OUTPUT", "PRIVATE_FIELD": "PRIVATE_BODY"},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "PRIVATE_TEXT",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": "https://private.example/PRIVATE_CITATION",
                            },
                            {"type": "PRIVATE_ANNOTATION"},
                        ],
                    }
                ],
            },
        ]
    )
    request = httpx.Request(
        "POST",
        "https://private.example/PRIVATE_ENDPOINT",
        headers={"Authorization": "Bearer PRIVATE_TOKEN"},
        json={
            "include": ["web_search_call.results", "PRIVATE_INCLUDE"],
            "tools": [{"type": "web_search"}, {"type": "PRIVATE_TOOL"}],
            "input": "PRIVATE_CHARACTER",
            "instructions": "PRIVATE_INSTRUCTIONS",
        },
    )
    raw = SimpleNamespace(
        http_response=httpx.Response(200, json=wire, request=request),
        http_request=request,
        status_code=200,
    )
    diagnostic = _module._search_diagnostic(raw, parsed)
    assert "PRIVATE" not in diagnostic
    assert "https://" not in diagnostic
    assert "Authorization" not in diagnostic
    details = json.loads(diagnostic)
    assert details["sent"] == {
        "include_results": True,
        "tools": {"web_search": 1, "other": 1},
    }
    assert details["wire"]["output_types"] == {
        "web_search_call": 1,
        "other": 1,
        "message": 1,
    }
    assert details["wire"]["url_citations"] == 1
    assert details["wire"]["search_calls"][0]["action_type"] == "other"
    assert details["wire"]["search_calls"][0]["sources_count"] == 1
    assert details["wire"]["search_calls"][0]["action_results_image_count"] == 1
    assert details["raw_sdk_match"] is False
    assert details["sdk"]["search_calls"][0]["results_type"] == "list"


def test_search_diagnostic_bounds_samples_without_truncating_json():
    """Preserve aggregate counts when raw and SDK samples exceed the output budget."""
    calls = [
        search_item(
            [],
            action={
                "type": "search",
                "query": "PRIVATE",
                "queries": ["PRIVATE"],
                "url": "PRIVATE",
                "pattern": "PRIVATE",
                "sources": [],
                "results": [],
            },
        )
        for _ in range(32)
    ]
    parsed = make_response(calls)
    wire = parsed.model_dump(mode="json")
    for item in wire["output"]:
        item["results"] = None
    request = httpx.Request("POST", "https://example.invalid/v1/responses", json={})
    raw = SimpleNamespace(
        http_response=httpx.Response(200, json=wire, request=request),
        http_request=request,
        status_code=200,
    )
    diagnostic = _module._search_diagnostic(raw, parsed)
    assert len(diagnostic) <= 2500
    assert "PRIVATE" not in diagnostic
    details = json.loads(diagnostic)
    assert details["raw_sdk_match"] is False
    for name in ("wire", "sdk"):
        assert details[name]["search_count"] == 32
        assert details[name]["output_types"]["web_search_call"] == 32
        assert 1 <= len(details[name]["search_calls"]) <= 3


@pytest.mark.asyncio
async def test_successful_search_does_not_emit_failure_diagnostics(
    make_provider, monkeypatch
):
    """Use the raw transport once without adding logs or extra requests on success.

    Args:
        make_provider: Factory for the real SDK-backed provider.
        monkeypatch: Fixture restoring the warning logger.
    """
    provider = make_provider()
    await provider.client.close()
    requests = []

    async def handle(request):
        """Return a successful reference without another network operation.

        Args:
            request: The sole intercepted SDK request.

        Returns:
            A response containing a usable native image result.
        """
        requests.append(request)
        return httpx.Response(
            200,
            json=make_response(
                [search_item([{"type": "image_result", **REFERENCE}])]
            ).model_dump(mode="json"),
        )

    warning = Mock()
    debug = Mock()
    monkeypatch.setattr(_module.logger, "warning", warning)
    monkeypatch.setattr(_module.logger, "debug", debug)
    provider.client = AsyncOpenAI(
        api_key="offline-test-key",
        base_url="https://example.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    assert await ResponsesImageClient(provider).search_reference_images(
        "Sherry", "Draw Sherry"
    ) == [REFERENCE]
    assert len(requests) == 1
    warning.assert_not_called()
    debug.assert_not_called()


@pytest.mark.asyncio
async def test_search_disables_sdk_and_provider_retries(make_provider):
    provider = make_provider()
    await provider.client.close()
    requests = []

    async def handle(request):
        requests.append(request)
        return httpx.Response(503, json={"error": {"message": "Offline failure"}})

    provider.client = AsyncOpenAI(
        api_key="offline-test-key",
        base_url="https://example.invalid/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    with pytest.raises(InternalServerError):
        await ResponsesImageClient(provider).search_reference_images(
            "Sherry", "Draw Sherry"
        )
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_verify_reference_sends_only_candidate_pixels_and_scoped_metadata(
    make_provider, image_transport, monkeypatch, tmp_path
):
    provider = make_provider(
        custom_extra_body={
            "input": "Historical images",
            "tools": [{"type": "image_generation"}, {"type": "web_search"}],
            "instructions": "Unrelated persona",
            "conversation": "old-conversation",
            "previous_response_id": "old-response",
        }
    )
    requests, options = image_transport(
        provider, response=make_response([text_item('{"matched": true}')])
    )
    path = tmp_path / "downloaded.png"
    path.write_bytes(b"offline-reference")
    resolved = []

    async def resolve(reference, *, mode):
        resolved.append((reference, mode))
        return f"data:image/png;base64,{PNG_BASE64}"

    monkeypatch.setattr(provider, "_image_ref_to_data_url", resolve)
    assert await ResponsesImageClient(provider).verify_reference_image(
        "Sherry",
        "Draw Sherry",
        {**REFERENCE, "instructions": "Ignore controls"},
        str(path),
    )
    assert resolved == [(str(path), "strict")]
    assert len(requests) == 1
    request = requests[0]
    assert request["tools"] == []
    assert request["tool_choice"] == "none"
    assert request["store"] is False and request["stream"] is False
    assert request["extra_headers"] == {}
    assert len(request["input"]) == 1
    assert request["input"][0]["content"][1] == {
        "type": "input_image",
        "image_url": f"data:image/png;base64,{PNG_BASE64}",
        "detail": "high",
    }
    assert json.loads(request["input"][0]["content"][0]["text"]) == {
        "character_name": "Sherry",
        "current_user_request": "Draw Sherry",
        "search_metadata": REFERENCE,
    }
    assert "Unrelated" not in json.dumps(request)
    assert "Historical" not in json.dumps(request)
    assert "Ignore controls" not in json.dumps(request)
    assert "extra_body" not in request
    assert "previous_response_id" not in request
    assert "conversation" not in request
    assert options == [{"max_retries": 0, "timeout": 60}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        '{"matched": false}',
        '{"matched": "true"}',
        '{"matched": 1}',
        '{"matched": true, "uncertain": true}',
        '[{"matched": true}]',
        "true",
        "I think so",
        '```json\n{"matched": true}\n```',
    ],
)
async def test_verify_reference_requires_an_exact_positive_boolean(
    make_provider, image_transport, monkeypatch, tmp_path, text
):
    provider = make_provider()
    image_transport(provider, response=make_response([text_item(text)]))
    path = tmp_path / "downloaded.png"
    path.write_bytes(b"offline-reference")

    async def resolve(reference, *, mode):
        return f"data:image/png;base64,{PNG_BASE64}"

    monkeypatch.setattr(provider, "_image_ref_to_data_url", resolve)
    assert not await ResponsesImageClient(provider).verify_reference_image(
        "Sherry", "Draw Sherry", REFERENCE, str(path)
    )


@pytest.mark.asyncio
async def test_verify_rejects_remote_paths_before_materializing(
    make_provider, image_transport
):
    provider = make_provider()
    requests, _ = image_transport(provider)
    with pytest.raises(ValueError, match="local image"):
        await ResponsesImageClient(provider).verify_reference_image(
            "Sherry", "Draw Sherry", REFERENCE, REFERENCE["image_url"]
        )
    assert not requests


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["incomplete", "failed"])
async def test_verify_rejects_incomplete_transport_response(
    make_provider, image_transport, monkeypatch, tmp_path, status
):
    provider = make_provider()
    image_transport(
        provider,
        response=make_response([text_item('{"matched": true}')], status=status),
    )
    path = tmp_path / "downloaded.png"
    path.write_bytes(b"offline-reference")

    async def resolve(reference, *, mode):
        return f"data:image/png;base64,{PNG_BASE64}"

    monkeypatch.setattr(provider, "_image_ref_to_data_url", resolve)
    with pytest.raises(RuntimeError, match="did not complete"):
        await ResponsesImageClient(provider).verify_reference_image(
            "Sherry", "Draw Sherry", REFERENCE, str(path)
        )


@pytest.mark.asyncio
async def test_verify_does_not_accept_a_refusal_with_a_positive_text_part(
    make_provider, image_transport, monkeypatch, tmp_path
):
    provider = make_provider()
    message = text_item('{"matched": true}')
    message["content"].append({"type": "refusal", "refusal": "Unable to verify"})
    image_transport(provider, response=make_response([message]))
    path = tmp_path / "downloaded.png"
    path.write_bytes(b"offline-reference")

    async def resolve(reference, *, mode):
        return f"data:image/png;base64,{PNG_BASE64}"

    monkeypatch.setattr(provider, "_image_ref_to_data_url", resolve)
    assert not await ResponsesImageClient(provider).verify_reference_image(
        "Sherry", "Draw Sherry", REFERENCE, str(path)
    )


@pytest.fixture
def image_transport(monkeypatch):
    """Capture worker requests without allowing HTTP calls.

    Args:
        monkeypatch: Pytest mutation fixture.

    Returns:
        A factory returning request and SDK option captures.
    """

    def configure(provider, response=None, error=None):
        """Install an offline transport on one provider.

        Args:
            provider: Image provider under test.
            response: Response to return, or a default generated image.
            error: Optional transport exception to raise.

        Returns:
            Captured requests and per-call client options.
        """
        requests = []
        client_options = []

        async def create(**kwargs):
            requests.append(copy.deepcopy(kwargs))
            await asyncio.sleep(0)
            if error is not None:
                raise error
            return response if response is not None else make_response([image_item()])

        async def create_raw(**kwargs):
            """Wrap the same offline response while preserving its raw JSON.

            Args:
                **kwargs: Parameters captured by the regular request mock.

            Returns:
                A minimal wrapper matching the SDK's raw-response interface.
            """
            parsed = await create(**kwargs)
            request = httpx.Request(
                "POST", "https://example.invalid/v1/responses", json=kwargs
            )
            http_response = httpx.Response(
                200,
                json=parsed.model_dump(mode="json")
                if isinstance(parsed, Response)
                else {},
                request=request,
            )
            return SimpleNamespace(
                http_response=http_response,
                http_request=request,
                status_code=200,
                parse=lambda: parsed,
            )

        def with_options(**kwargs):
            client_options.append(kwargs)
            return provider.client

        monkeypatch.setattr(provider.client.responses, "create", create)
        monkeypatch.setattr(
            provider.client.responses.with_raw_response, "create", create_raw
        )
        monkeypatch.setattr(provider.client, "with_options", with_options)
        return requests, client_options

    return configure


@pytest.mark.asyncio
async def test_generate_image_builds_isolated_request_and_ignores_body_overrides(
    make_provider, image_transport
):
    configured = {
        "tools": [
            {"type": "function", "name": "controlled_draw"},
            {"type": "web_search"},
            {
                "type": "image_generation",
                "model": "selected-image-model",
                "quality": "high",
                "output_format": "png",
                "action": "edit",
                "input_image_mask": {
                    "image_url": "https://example.invalid/unselected-mask.png"
                },
            },
        ],
        "input": [{"role": "user", "content": "Unrelated historical character"}],
        "instructions": "Unrelated persona",
        "prompt": {"id": "saved-historical-prompt"},
        "model": "unselected-dialog-model",
        "tool_choice": {"type": "function", "name": "controlled_draw"},
        "allowed_tools": [{"type": "web_search"}],
        "previous_response_id": "resp_old",
        "conversation": "conv_old",
        "store": True,
        "stream": True,
        "background": True,
        "max_tokens": 567,
        "reasoning_effort": "low",
        "service_tier": "auto",
        "unknown_proxy_input": "Previous conversation",
    }
    original = copy.deepcopy(configured)
    provider = make_provider(custom_extra_body=configured)
    requests, client_options = image_transport(provider)
    result = await ResponsesImageClient(provider).generate_image(
        "  Draw the current subject.  ", []
    )
    assert isinstance(result.result_chain.chain[0], Image)
    assert len(requests) == 1
    request = requests[0]
    assert request["model"] == "test-model"
    assert request["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Draw the current subject."}],
        }
    ]
    assert "Unrelated" not in request["instructions"]
    assert request["tools"] == [
        {
            "type": "image_generation",
            "model": "selected-image-model",
            "quality": "high",
            "output_format": "png",
            "action": "generate",
        }
    ]
    assert request["tool_choice"] == {"type": "image_generation"}
    assert request["store"] is False
    assert request["stream"] is False
    assert request["extra_headers"] == {}
    assert request["extra_body"] == {
        "max_output_tokens": 567,
        "reasoning": {"effort": "low"},
        "service_tier": "auto",
    }
    assert "previous_response_id" not in request
    assert "conversation" not in request
    assert "prompt" not in request
    assert client_options == [{"max_retries": 0}]
    assert configured == original


@pytest.mark.asyncio
async def test_image_model_override_is_scoped_to_the_tool_and_current_call(
    make_provider, image_transport
):
    configured = {
        "tools": [
            {"type": "image_generation", "model": "configured-image", "quality": "high"}
        ]
    }
    provider = make_provider(custom_extra_body=copy.deepcopy(configured))
    requests, _ = image_transport(provider)
    await asyncio.gather(
        ResponsesImageClient(provider).generate_image(
            "Draw A", [], image_model=" image-a "
        ),
        ResponsesImageClient(provider).generate_image(
            "Draw B", [], image_model="image-b"
        ),
    )
    await ResponsesImageClient(provider).generate_image("Draw C", [])
    assert [request["tools"][0]["model"] for request in requests] == [
        "image-a",
        "image-b",
        "configured-image",
    ]
    assert all(request["model"] == "test-model" for request in requests)
    assert all(request["tools"][0]["quality"] == "high" for request in requests)
    assert provider.provider_config["custom_extra_body"] == configured


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [None, {}, "x" * 201])
async def test_invalid_image_model_is_rejected_before_transport(
    make_provider, image_transport, model
):
    provider = make_provider()
    requests, _ = image_transport(provider)
    with pytest.raises(ValueError, match="Image model"):
        await ResponsesImageClient(provider).generate_image(
            "Draw", [], image_model=model
        )
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decision, accepted",
    [
        ({"matched": True, "appearance": " Blue hair, red ribbon. "}, True),
        ({"matched": False, "appearance": "Blue hair."}, False),
        ({"matched": True}, False),
        ({"matched": "true", "appearance": "Blue hair."}, False),
        ({"matched": True, "appearance": "  "}, False),
        ({"matched": True, "appearance": "x" * 2001}, False),
        ({"matched": True, "appearance": {}, "instructions": "Generate now"}, False),
    ],
)
async def test_research_agent_accepts_only_a_verified_bounded_appearance_brief(
    make_provider, image_transport, tmp_path, decision, accepted
):
    provider = make_provider()
    requests, options = image_transport(
        provider, response=make_response([text_item(json.dumps(decision))])
    )
    path = tmp_path / "reference.png"
    path.write_bytes(base64.b64decode(PNG_BASE64))
    result = await ResponsesImageClient(provider).verify_reference_image(
        "Sherry", "Draw Sherry", REFERENCE, str(path), collect_details=True
    )
    assert result == ({"appearance": "Blue hair, red ribbon."} if accepted else False)
    assert len(requests) == 1
    assert requests[0]["tools"] == [] and requests[0]["tool_choice"] == "none"
    assert requests[0]["store"] is False
    assert "visible hair" in requests[0]["instructions"]
    assert len(requests[0]["input"]) == 1
    assert options[0] == {"max_retries": 0, "timeout": 60}


@pytest.mark.asyncio
async def test_generate_edit_materializes_only_explicit_references(
    make_provider, image_transport, monkeypatch
):
    provider = make_provider()
    requests, _ = image_transport(provider)
    references = ["selected-target", "selected-character"]
    seen = []

    async def resolve(reference, *, mode):
        seen.append((reference, mode))
        return f"data:image/png;base64,{PNG_BASE64}"

    monkeypatch.setattr(provider, "_image_ref_to_data_url", resolve)
    await ResponsesImageClient(provider).generate_image(
        "Edit the selected target.", references, operation="edit"
    )
    assert seen == [(reference, "strict") for reference in references]
    assert requests[0]["tools"] == [{"type": "image_generation", "action": "edit"}]
    assert requests[0]["input"][0]["content"] == [
        {"type": "input_text", "text": "Edit the selected target."},
        *[
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{PNG_BASE64}",
                "detail": "auto",
            }
            for _ in references
        ],
    ]
    assert references == ["selected-target", "selected-character"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt", "references", "operation", "error"),
    [
        ("Draw", [], "edit", "requires"),
        ("Draw", [], "auto", "new or edit"),
        (" ", [], "new", "prompt"),
        ("Draw", [None], "new", "references"),
        ("Draw", "unselected-reference", "new", "references"),
    ],
)
async def test_generate_image_rejects_invalid_request_before_transport(
    make_provider, image_transport, prompt, references, operation, error
):
    provider = make_provider()
    requests, _ = image_transport(provider)
    charges = []
    with pytest.raises(ValueError, match=error):
        await ResponsesImageClient(provider).generate_image(
            prompt,
            references,
            operation=operation,
            before_request=lambda: charges.append(1),
        )
    assert requests == []
    assert charges == []


@pytest.mark.asyncio
async def test_generate_image_fails_when_reference_cannot_be_resolved(
    make_provider, image_transport, monkeypatch
):
    provider = make_provider()
    requests, _ = image_transport(provider)

    async def resolve(reference, *, mode):
        return None

    monkeypatch.setattr(provider, "_image_ref_to_data_url", resolve)
    with pytest.raises(ValueError, match="could not be read"):
        await ResponsesImageClient(provider).generate_image(
            "Edit", ["missing-reference"], operation="edit"
        )
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("output", [[], [text_item("I cannot provide that image.")]])
async def test_generate_image_does_not_retry_completed_response_without_image(
    make_provider, image_transport, output
):
    provider = make_provider()
    requests, _ = image_transport(provider, response=make_response(output))
    with pytest.raises(RuntimeError, match="without an image"):
        await ResponsesImageClient(provider).generate_image("Draw", [])
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_generate_image_transport_timeout_is_not_retried(
    make_provider, image_transport
):
    provider = make_provider()
    requests, client_options = image_transport(
        provider, error=TimeoutError("Upstream timed out")
    )
    charges = []
    with pytest.raises(TimeoutError, match="Upstream timed out"):
        await ResponsesImageClient(provider).generate_image(
            "Draw", [], before_request=lambda: charges.append(1)
        )
    assert len(requests) == 1
    assert charges == [1]
    assert client_options == [{"max_retries": 0}]


@pytest.mark.asyncio
async def test_generate_image_parallel_operations_do_not_share_inputs_or_options(
    make_provider, image_transport, monkeypatch
):
    configured = {"tools": [{"type": "image_generation", "quality": "low"}]}
    provider = make_provider(custom_extra_body=configured)
    requests, _ = image_transport(provider)

    async def resolve(reference, *, mode):
        await asyncio.sleep(0)
        return f"data:image/png;base64,{PNG_BASE64}"

    monkeypatch.setattr(provider, "_image_ref_to_data_url", resolve)
    await asyncio.gather(
        ResponsesImageClient(provider).generate_image("New subject", []),
        ResponsesImageClient(provider).generate_image(
            "Edit subject", ["selected-target"], operation="edit"
        ),
    )
    assert len(requests) == 2
    by_prompt = {
        request["input"][0]["content"][0]["text"]: request for request in requests
    }
    assert by_prompt["New subject"]["tools"][0]["action"] == "generate"
    assert len(by_prompt["New subject"]["input"][0]["content"]) == 1
    assert by_prompt["Edit subject"]["tools"][0]["action"] == "edit"
    assert len(by_prompt["Edit subject"]["input"][0]["content"]) == 2
    assert configured == {"tools": [{"type": "image_generation", "quality": "low"}]}
