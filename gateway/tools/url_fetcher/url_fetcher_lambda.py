# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Fetch the readable text of a public web page.

Search tools return snippets plus a URL, so without this the agent cites documents it
has never read. That inflates apparent grounding: a claim can be attributed to a real
URL whose content nobody checked.

The model chooses the URL, which makes this a server-side request forgery primitive if
left open. Requests are therefore resolved and validated before connecting, and every
redirect hop is validated the same way -- a public hostname that redirects to
169.254.169.254 would otherwise reach the instance metadata endpoint.
"""

import html
import ipaddress
import logging
import re
import socket
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MAX_BYTES = 400_000
MAX_CHARS = 20_000
TIMEOUT_SECONDS = 15
MAX_REDIRECTS = 3
UA = "Mozilla/5.0 (compatible; AgentCoreDeepResearch/1.0)"

# text/* is allowed as a prefix; these are the non-text types worth reading.
ALLOWED_CONTENT = ("text/", "application/json", "application/xml", "application/xhtml+xml")

DROP_TAGS = re.compile(
    r"<(script|style|noscript|svg|nav|footer|header|form|aside)\b.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
BLOCK_BOUNDARY = re.compile(r"</(p|div|h[1-6]|li|tr|section|article|blockquote)\s*>", re.IGNORECASE)
TAG = re.compile(r"<[^>]+>")
BLANK_RUN = re.compile(r"\n{3,}")


def _assert_public(hostname: str) -> None:
    """
    Resolve a hostname and reject any address that is not publicly routable.

    Checked per redirect hop rather than once, since the guarantee needed is about the
    address actually connected to, not the one originally requested.
    """
    if not hostname:
        raise ValueError("URL has no hostname")
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"cannot resolve {hostname}") from exc

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        # link_local covers 169.254.0.0/16, which is the cloud metadata endpoint.
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError(f"refusing to fetch non-public address {ip} for {hostname}")


def _validate(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"only http and https are supported, got {parsed.scheme!r}")
    _assert_public(parsed.hostname or "")
    return url


def html_to_text(raw: str) -> str:
    """Strip markup to readable text, keeping block structure as line breaks."""
    text = DROP_TAGS.sub(" ", raw)
    text = BLOCK_BOUNDARY.sub("\n", text)
    text = TAG.sub(" ", text)
    text = html.unescape(text)
    lines = [re.sub(r"[ \t\u00a0]+", " ", ln).strip() for ln in text.split("\n")]
    return BLANK_RUN.sub("\n\n", "\n".join(ln for ln in lines if ln))


def fetch_url(url: str, max_chars: int = MAX_CHARS) -> str:
    """
    Fetch `url` and return its readable text, truncated to `max_chars`.

    Redirects are followed manually so each hop can be revalidated.
    """
    current = _validate(url)
    for _ in range(MAX_REDIRECTS + 1):
        request = urllib.request.Request(  # noqa: S310 - scheme and address validated above
            current, headers={"User-Agent": UA, "Accept": "text/html,text/plain,*/*"}
        )
        opener = urllib.request.build_opener(_NoRedirect)
        # _NoRedirect makes urllib raise on 3xx instead of following, so the redirect
        # arrives as an HTTPError -- which is itself a readable response carrying Location.
        try:
            response = opener.open(request, timeout=TIMEOUT_SECONDS)
        except HTTPError as exc:
            if exc.code not in (301, 302, 303, 307, 308):
                raise
            location = exc.headers.get("Location")
            if not location:
                raise ValueError(f"redirect from {current} with no Location header") from exc
            current = _validate(urllib.parse.urljoin(current, location))
            continue

        content_type = (response.headers.get("Content-Type") or "").lower()
        if content_type and not content_type.startswith(ALLOWED_CONTENT):
            raise ValueError(f"unsupported content type {content_type.split(';')[0]}")

        raw = response.read(MAX_BYTES).decode(
            response.headers.get_content_charset() or "utf-8", errors="replace"
        )
        text = html_to_text(raw) if "html" in content_type or raw.lstrip().startswith("<") else raw
        truncated = len(text) > max_chars
        body = text[:max_chars] + ("\n\n[truncated]" if truncated else "")
        return f"Source: {current}\n\n{body}"

    raise ValueError(f"too many redirects starting at {url}")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Surface redirects to the caller so the destination can be revalidated."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def handler(event, context):
    """Gateway entry point."""
    tool_name = event.get("tool_name") or (
        context.client_context.custom.get("bedrockAgentCoreToolName", "")
        if context and getattr(context, "client_context", None)
        else ""
    )
    if tool_name and "fetch_url" not in tool_name:
        return {"error": f"This Lambda only supports 'fetch_url', received: {tool_name}"}

    url = (event.get("url") or "").strip()
    if not url:
        return {"error": "url is required"}
    max_chars = event.get("max_chars", MAX_CHARS)
    if not isinstance(max_chars, int) or max_chars < 500:
        max_chars = MAX_CHARS
    max_chars = min(max_chars, MAX_CHARS)

    try:
        return {"content": [{"type": "text", "text": fetch_url(url, max_chars)}]}
    except (HTTPError, URLError) as exc:
        return {"error": f"Could not fetch {url}: {exc}"}
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - a tool error must not kill the rollout
        logger.error(f"Unexpected error fetching {url}: {exc}")
        return {"error": f"Could not fetch {url}: {exc}"}
