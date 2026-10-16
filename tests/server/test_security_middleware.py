import asyncio
import sys
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from httpx import ASGITransport, AsyncClient

from private_gpt.launcher import create_app
from private_gpt.server.utils.security import uvicorn_proxy_kwargs
from private_gpt.settings.settings import SecuritySettings
from tests.fixtures.mock_injector import MockInjector

CORS = {
    "enabled": True,
    "allow_origins": ["https://app.example"],
    "allow_methods": ["GET", "POST"],
    "allow_headers": ["*"],
}
ORIGIN = {"Origin": "https://app.example"}


def _security(**overrides: Any) -> dict[str, Any]:
    return {"server": {"cors": CORS, "security": {"enabled": True, **overrides}}}


@pytest.fixture
def make_app(applied_migrations: MockInjector) -> Callable[..., FastAPI]:
    def _make(settings: dict[str, Any]) -> FastAPI:
        applied_migrations.bind_settings(settings)
        return create_app(applied_migrations.test_injector)

    return _make


def _client(app: FastAPI, ip: str) -> AsyncClient:
    return AsyncClient(
        transport=ASGITransport(app=app, client=(ip, 50000)),
        base_url="http://test",
    )


async def test_disabled_adds_nothing_and_never_imports_guard(
    make_app: Callable[..., FastAPI], monkeypatch: pytest.MonkeyPatch
) -> None:
    # A None entry makes `import guard` raise ImportError.
    monkeypatch.setitem(sys.modules, "guard", None)
    app = make_app({"server": {"cors": CORS}})

    assert [m.cls.__name__ for m in app.user_middleware].count(
        "PureASGISecurityMiddleware"
    ) == 0
    async with _client(app, "203.0.113.1") as client:
        for _ in range(200):
            assert (await client.get("/health")).status_code == 200


def test_enabled_without_fastapi_guard_raises_clear_error(
    make_app: Callable[..., FastAPI], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "guard", None)
    with pytest.raises(RuntimeError, match=r"private-gpt\[security\]"):
        make_app(_security())


async def test_burst_returns_429_with_cors_headers(
    make_app: Callable[..., FastAPI],
) -> None:
    app = make_app(_security(rate_limit=4, rate_limit_window=60, exclude_paths=[]))
    async with _client(app, "203.0.113.2") as client:
        codes = [
            (await client.get("/v1/models", headers=ORIGIN)).status_code
            for _ in range(3)
        ]
        assert 429 not in codes
        assert (
            "x-frame-options"
            not in (await client.get("/v1/models", headers=ORIGIN)).headers
        )
        limited = await client.get("/v1/models", headers=ORIGIN)

    assert limited.status_code == 429
    # CORS sits outside the guard, so the browser can read the rejection.
    assert limited.headers["access-control-allow-origin"] == "https://app.example"


async def test_allowlist_rejects_other_ips_with_403(
    make_app: Callable[..., FastAPI],
) -> None:
    app = make_app(_security(whitelist=["198.51.100.7"], exclude_paths=[]))
    async with _client(app, "203.0.113.3") as outsider:
        denied = await outsider.get("/v1/models", headers=ORIGIN)
    async with _client(app, "198.51.100.7") as insider:
        allowed = await insider.get("/v1/models")

    assert denied.status_code == 403
    assert denied.headers["access-control-allow-origin"] == "https://app.example"
    assert allowed.status_code != 403


async def test_spoofed_forwarded_for_from_untrusted_client_is_ignored(
    make_app: Callable[..., FastAPI],
) -> None:
    app = make_app(_security(whitelist=["198.51.100.8"], exclude_paths=[]))
    async with _client(app, "203.0.113.4") as client:
        response = await client.get(
            "/v1/models", headers={"X-Forwarded-For": "198.51.100.8"}
        )

    assert response.status_code == 403


async def test_forwarded_for_from_trusted_proxy_is_honoured(
    make_app: Callable[..., FastAPI],
) -> None:
    app = make_app(_security(whitelist=["198.51.100.9"], exclude_paths=[]))
    async with _client(app, "127.0.0.1") as proxy:
        response = await proxy.get(
            "/v1/models", headers={"X-Forwarded-For": "198.51.100.9"}
        )

    assert response.status_code != 403


def test_uvicorn_proxy_kwargs_only_change_anything_when_enabled() -> None:
    assert uvicorn_proxy_kwargs(SecuritySettings()) == {}
    assert uvicorn_proxy_kwargs(SecuritySettings(enabled=True, trusted_proxies=[])) == {
        "proxy_headers": False
    }
    assert uvicorn_proxy_kwargs(
        SecuritySettings(enabled=True, trusted_proxies=["10.0.0.0/8"])
    ) == {"proxy_headers": True, "forwarded_allow_ips": ["10.0.0.0/8"]}


async def test_uvicorn_cannot_hand_the_guard_a_forwarded_address(
    make_app: Callable[..., FastAPI], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Uvicorn would otherwise trust X-Forwarded-For from any peer.
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "*")
    security = {"rate_limit": 2, "rate_limit_window": 60, "exclude_paths": []}
    app = make_app(_security(**security))
    config = uvicorn.Config(
        app,
        log_config=None,
        **uvicorn_proxy_kwargs(SecuritySettings(enabled=True, **security)),
    )
    config.load()

    async with _client(config.loaded_app, "203.0.113.9") as client:
        codes = [
            (
                await client.get(
                    "/v1/models", headers={"X-Forwarded-For": f"198.18.0.{i}"}
                )
            ).status_code
            for i in range(4)
        ]

    assert codes[2:] == [429, 429]


async def test_attack_in_query_is_blocked(make_app: Callable[..., FastAPI]) -> None:
    app = make_app(_security(exclude_paths=[]))
    async with _client(app, "203.0.113.5") as client:
        response = await client.get(
            "/v1/models", params={"q": "1' UNION SELECT password FROM users--"}
        )

    assert response.status_code == 400


@pytest.mark.parametrize(
    "referer",
    [
        "http://localhost:8001/",
        "http://127.0.0.1:8001/",
        "http://192.168.1.20:8001/",
    ],
)
async def test_localhost_and_lan_referers_are_not_blocked(
    make_app: Callable[..., FastAPI], referer: str
) -> None:
    app = make_app(_security(exclude_paths=[]))
    async with _client(app, f"203.0.113.{10 + len(referer)}") as client:
        response = await client.get("/v1/models", headers={"Referer": referer})

    assert response.status_code != 400


async def test_attack_in_referer_is_still_blocked(
    make_app: Callable[..., FastAPI],
) -> None:
    app = make_app(_security(exclude_paths=[]))
    async with _client(app, "203.0.113.20") as client:
        response = await client.get(
            "/v1/models", headers={"Referer": "<script>alert(1)</script>"}
        )

    assert response.status_code == 400


async def test_body_is_not_scanned_by_default(
    make_app: Callable[..., FastAPI],
) -> None:
    payload = {"input": "<script>alert(1)</script> ' OR 1=1 --"}

    app = make_app(_security(exclude_paths=[]))
    async with _client(app, "203.0.113.6") as client:
        default = await client.post("/v1/embeddings", json=payload)
    app = make_app(_security(exclude_paths=[], scan_body=True))
    async with _client(app, "203.0.113.7") as client:
        scanned = await client.post("/v1/embeddings", json=payload)

    # The route reached the app (not the guard) when the body is not scanned.
    assert "Suspicious activity" not in default.text
    assert scanned.status_code == 400
    assert "Suspicious activity" in scanned.text


async def test_streaming_is_delivered_incrementally(
    make_app: Callable[..., FastAPI],
) -> None:
    app = make_app(_security(exclude_paths=[]))
    first_chunk_seen = asyncio.Event()

    async def chunks() -> AsyncIterator[bytes]:
        yield b"data: one\n\n"
        # Only continues once the client side saw chunk one, which is
        # impossible if a middleware buffers the response body.
        await asyncio.wait_for(first_chunk_seen.wait(), timeout=5)
        yield b"data: two\n\n"

    @app.get("/_test_stream")
    async def stream() -> StreamingResponse:
        return StreamingResponse(chunks(), media_type="text/event-stream")

    bodies: list[bytes] = []

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            bodies.append(message["body"])
            first_chunk_seen.set()

    async def receive() -> dict[str, Any]:
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "path": "/_test_stream",
        "raw_path": b"/_test_stream",
        "query_string": b"",
        "root_path": "",
        "scheme": "http",
        "headers": [(b"host", b"test")],
        "client": ("203.0.113.8", 50000),
        "server": ("test", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), timeout=10)

    assert bodies == [b"data: one\n\n", b"data: two\n\n"]
