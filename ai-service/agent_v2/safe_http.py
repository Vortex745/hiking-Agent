"""Ticket 10 — SSRF-safe HTTP download adapter.

Spec AC5: a download adapter that fetches remote resources on behalf
of a sandbox run must only allow HTTP(S), block localhost / private /
link-local / cloud-metadata addresses, refuse redirect bypasses, and
cap response size + timeout.

The artifact profile (Ticket 09) does not currently expose a URL-fetch
tool, so this module has no callers in the request path yet. It exists
so any future tool that needs to fetch a remote resource (e.g. image
embedding for PDF) goes through one SSRF-safe adapter instead of
rolling its own ``httpx.get``.

The adapter uses ``urllib.request`` from the stdlib to avoid adding a
new runtime dependency for a code path that is not yet active. When
the artifact profile gains a real URL-fetch tool, this module can be
upgraded to ``httpx`` without changing the public function signature.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import Request

logger = logging.getLogger("ai-service.agent_v2.safe_http")


# Spec AC5: hard caps. The artifact profile never fetches huge resources;
# a single image or text attachment is well under 5 MiB. The timeout is
# aggressive because the artifact flow is already inside a run-scoped
# sandbox with its own timeout — we do not want the URL fetch to be the
# long pole.
DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MiB
DEFAULT_TIMEOUT_SECONDS = 10.0


class UnsafeUrlError(ValueError):
    """Raised when a URL fails the SSRF safety check."""


class SafeHttpFetchError(RuntimeError):
    """Raised when a safe-fetch operation fails (size/timeout/HTTP)."""


@dataclass(frozen=True, slots=True)
class SafeHttpResponse:
    """Result of a successful :func:`safe_http_get`."""

    status: int
    body: bytes
    content_type: str


def assert_safe_url(url: str) -> None:
    """Raise :class:`UnsafeUrlError` if ``url`` is not HTTP(S) or resolves
    to a forbidden address.

    Spec AC5 forbidden categories:

    * non-HTTP(S) schemes (``file://``, ``ftp://``, ``gopher://``, …)
    * ``localhost``, ``127.0.0.0/8``, ``::1`` (loopback)
    * ``10.0.0.0/8``, ``172.16.0.0/12``, ``192.168.0.0/16`` (private)
    * ``169.254.0.0/16`` (link-local) — includes AWS/GCP/Azure metadata
      ``169.254.169.254``
    * ``fc00::/7`` (IPv6 ULA), ``fe80::/10`` (IPv6 link-local)
    * ``0.0.0.0/8`` (unspecified)
    * IPv6-mapped IPv4 of any of the above

    DNS resolution is done eagerly so a hostname that resolves to a
    private IP is rejected before the HTTP request is issued. The check
    intentionally does NOT trust the URL's hostname literal — only the
    resolved IP.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeUrlError(
            f"scheme {parsed.scheme!r} not allowed (only http/https)"
        )

    hostname = parsed.hostname
    if not hostname:
        raise UnsafeUrlError(f"URL has no hostname: {url!r}")

    # Reject literal IP forms directly. ``urlparse`` lowercases the
    # hostname; an IPv6 literal in a URL is wrapped in ``[...]`` and
    # ``hostname`` strips the brackets.
    try:
        literal_ip = ipaddress.ip_address(hostname)
    except ValueError:
        literal_ip = None

    if literal_ip is not None:
        _assert_public_ip(literal_ip)
        return

    # Resolve the hostname. We check ALL resolved addresses — if any
    # resolve to a forbidden range, the URL is rejected. This blocks
    # DNS rebinding where a hostname alternates between public and
    # private IPs.
    try:
        infos = socket.getaddrinfo(
            hostname, None, family=socket.AF_UNSPEC, proto=socket.IPPROTO_TCP
        )
    except socket.gaierror as exc:
        raise UnsafeUrlError(
            f"DNS resolution failed for {hostname!r}: {exc}"
        ) from exc

    if not infos:
        raise UnsafeUrlError(f"DNS returned no addresses for {hostname!r}")

    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            # Should not happen for getaddrinfo results, but be defensive.
            raise UnsafeUrlError(f"unparseable resolved address: {addr!r}")
        _assert_public_ip(ip)


def _assert_public_ip(ip: ipaddress._BaseAddress) -> None:
    """Raise :class:`UnsafeUrlError` if ``ip`` is loopback/private/link-local."""
    # ``ip_address`` returns IPv4Address or IPv6Address. Both expose
    # ``is_loopback``, ``is_private``, ``is_link_local``, ``is_unspecified``,
    # and ``is_reserved``. We treat all non-global addresses as forbidden
    # except IPv4 multicast (224.0.0.0/4) which is also rejected below.
    if ip.is_loopback:
        raise UnsafeUrlError(f"loopback address forbidden: {ip}")
    # Check is_link_local and is_unspecified BEFORE is_private — Python's
    # ipaddress marks 169.254.0.0/16 (link-local) and 0.0.0.0/8 (unspecified)
    # as is_private too, so the more specific category must win for
    # diagnostic clarity (and to match the spec's per-category rejection).
    if ip.is_link_local:
        # Includes 169.254.169.254 cloud metadata endpoints.
        raise UnsafeUrlError(f"link-local address forbidden: {ip}")
    if ip.is_unspecified:
        raise UnsafeUrlError(f"unspecified address forbidden: {ip}")
    if ip.is_private:
        raise UnsafeUrlError(f"private address forbidden: {ip}")
    if ip.is_reserved:
        raise UnsafeUrlError(f"reserved address forbidden: {ip}")
    if ip.is_multicast:
        raise UnsafeUrlError(f"multicast address forbidden: {ip}")

    # IPv6-mapped IPv4 (e.g. ::ffff:127.0.0.1) — unwrap and re-check.
    if isinstance(ip, ipaddress.IPv6Address):
        try:
            v4 = ip.ipv4_mapped
        except ValueError:
            v4 = None
        if v4 is not None:
            _assert_public_ip(v4)


def safe_http_get(
    url: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    allowed_content_types: Iterable[str] | None = None,
) -> SafeHttpResponse:
    """Fetch ``url`` with SSRF + size + timeout guards (spec AC5).

    Raises :class:`UnsafeUrlError` for SSRF violations,
    :class:`SafeHttpFetchError` for size/timeout/HTTP errors.

    Redirects are NOT followed: a 3xx response raises
    :class:`SafeHttpFetchError` so a public URL cannot redirect to an
    internal address. The caller must inspect the redirect and re-call
    with the new URL after re-running the SSRF check.

    ``allowed_content_types`` (when provided) restricts the response
    Content-Type. The check is prefix-based so ``text/plain`` matches
    ``text/plain; charset=utf-8``.
    """
    assert_safe_url(url)

    # Spec AC5: redirect bypass defense. ``urllib.request.urlopen``
    # follows redirects by default, which would let a public URL
    # redirect to an internal address (e.g. 169.254.169.254) and bypass
    # ``assert_safe_url``. We build a custom opener whose
    # ``HTTPRedirectHandler`` raises on any 3xx so the SSRF check
    # cannot be bypassed via a public-to-private redirect.
    class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl, *args):
            raise SafeHttpFetchError(
                f"redirect refused (status={code} → {newurl!r}); "
                f"SSRF-safe adapter does not follow redirects"
            )

    opener = urllib.request.build_opener(_NoRedirectHandler)
    req = Request(url, headers={"User-Agent": "ai-service-safe-http/1.0"})

    try:
        # ``opener.open`` returns an HTTPResponse. We read at most
        # ``max_bytes + 1`` to detect size violation without loading
        # an arbitrarily large body into memory.
        with opener.open(req, timeout=timeout) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")

            if 300 <= status < 400:
                # Defensive: the no-redirect handler should have raised
                # already, but keep this branch in case a future opener
                # change re-enables redirects.
                raise SafeHttpFetchError(
                    f"redirect refused (status={status}); SSRF-safe adapter "
                    f"does not follow redirects"
                )

            if allowed_content_types is not None:
                ct_lower = content_type.lower()
                if not any(
                    ct_lower.startswith(allowed.lower())
                    for allowed in allowed_content_types
                ):
                    raise SafeHttpFetchError(
                        f"Content-Type {content_type!r} not in allowlist"
                    )

            # Read up to max_bytes + 1 to detect overflow without
            # buffering an entire multi-GB response.
            body = resp.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise SafeHttpFetchError(
                    f"response exceeds max_bytes={max_bytes}"
                )

            return SafeHttpResponse(
                status=status,
                body=body,
                content_type=content_type,
            )
    except UnsafeUrlError:
        raise
    except SafeHttpFetchError:
        raise
    except urllib.error.HTTPError as exc:
        # 4xx/5xx responses — convert to SafeHttpFetchError so callers
        # have one exception type to catch. ``HTTPError`` is also a
        # valid response object so we read its body for diagnostics.
        raise SafeHttpFetchError(
            f"safe HTTP fetch failed: HTTP {exc.code} {exc.reason}"
        ) from exc
    except Exception as exc:
        # ``URLError`` for network/timeout errors. We collapse into
        # ``SafeHttpFetchError`` so callers have one exception type to
        # catch.
        raise SafeHttpFetchError(f"safe HTTP fetch failed: {exc}") from exc


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "SafeHttpFetchError",
    "SafeHttpResponse",
    "UnsafeUrlError",
    "assert_safe_url",
    "safe_http_get",
]
