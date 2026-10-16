"""Optional request security middleware, backed by fastapi-guard.

Everything here is opt-in: nothing is imported unless ``server.security.enabled``
is true, so installs without the ``security`` extra are unaffected.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

    from private_gpt.settings.settings import SecuritySettings


def install_security_middleware(app: "FastAPI", settings: "SecuritySettings") -> None:
    """Mount the pure ASGI fastapi-guard middleware on the app."""
    try:
        from guard import PureASGISecurityMiddleware, SecurityConfig
        from guard_core.handlers.suspatterns_handler import ALL_DETECTION_CATEGORIES
    except ImportError as e:
        raise RuntimeError(
            "server.security.enabled is true but fastapi-guard is not installed. "
            'Install it with: pip install "private-gpt[security]"'
        ) from e

    config = SecurityConfig(
        trusted_proxies=tuple(settings.trusted_proxies),
        trusted_proxy_depth=settings.trusted_proxy_depth,
        enable_rate_limiting=True,
        rate_limit=settings.rate_limit,
        rate_limit_window=settings.rate_limit_window,
        whitelist=tuple(settings.whitelist) if settings.whitelist is not None else None,
        blacklist=tuple(settings.blacklist),
        enable_penetration_detection=settings.penetration_detection,
        detection_scan_body=settings.scan_body,
        # Browsers send the page address as Referer, and guard-core 4.4.0 flags
        # localhost and LAN addresses in it as ssrf.
        enabled_detection_categories=frozenset(ALL_DETECTION_CATEGORIES) - {"ssrf"},
        exclude_paths=settings.exclude_paths,
        # Not part of this feature: no extra response headers (HSTS, frame options)
        security_headers={"enabled": False},
        # CORS stays with Starlette's CORSMiddleware, mounted outside the guard
        enable_cors=False,
        # In-memory state, no telemetry
        enable_redis=False,
        enable_agent=False,
    )
    app.add_middleware(PureASGISecurityMiddleware, config=config)


def uvicorn_proxy_kwargs(settings: "SecuritySettings") -> dict[str, Any]:
    """Uvicorn options that trust the same proxies as the security middleware.

    Uvicorn rewrites the client address from X-Forwarded-For (FORWARDED_ALLOW_IPS)
    before the app runs, so it must not trust more proxies than the guard does.
    """
    if not settings.enabled:
        return {}
    if not settings.trusted_proxies:
        return {"proxy_headers": False}
    return {
        "proxy_headers": True,
        "forwarded_allow_ips": list(settings.trusted_proxies),
    }
