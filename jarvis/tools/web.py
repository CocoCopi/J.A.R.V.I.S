"""Web tools: fetch a page, search the web, open a URL on the device.

No API keys required: search uses DuckDuckGo's HTML endpoint and results are
converted to plain text, which keeps token usage low on a phone data plan.
"""

from __future__ import annotations

import html
import re
from typing import List
from urllib.parse import quote_plus, urlparse

import requests

from .base import ToolResult, tool

USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 13; Termux) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Mobile Safari/537.36 Jarvis/0.3"
)
TIMEOUT = 25
MAX_TEXT = 12_000

_SCRIPT_STYLE = re.compile(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", re.S | re.I)
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n{3,}")


def html_to_text(markup: str) -> str:
    """Convert HTML to readable plain text."""
    text = _SCRIPT_STYLE.sub(" ", markup)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(p|div|li|tr|h[1-6])>", "\n", text, flags=re.I)
    text = _TAGS.sub(" ", text)
    text = html.unescape(text)
    text = _WS.sub(" ", text)
    text = _BLANKS.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _normalise_url(url: str) -> str:
    url = url.strip()
    if not url:
        return url
    if not urlparse(url).scheme:
        url = "https://" + url
    return url


@tool(category="web")
def web_fetch(url: str, max_chars: int = MAX_TEXT) -> ToolResult:
    """Download a web page and return it as plain text.

    Args:
        url: Address to fetch (https:// is assumed when missing).
        max_chars: Maximum number of characters to return.
    """
    target = _normalise_url(url)
    if not target:
        return ToolResult(output="no URL provided", ok=False)
    try:
        response = requests.get(
            target,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"},
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        return ToolResult(output="request failed: %s" % exc, ok=False)
    if response.status_code >= 400:
        return ToolResult(
            output="HTTP %d for %s" % (response.status_code, target), ok=False
        )
    content_type = response.headers.get("Content-Type", "")
    if "html" in content_type or "<html" in response.text[:500].lower():
        text = html_to_text(response.text)
    else:
        text = response.text
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated, %d chars total]" % len(text)
    return ToolResult(output=text or "(empty response)", ok=True)


@tool(category="web")
def web_search(query: str, max_results: int = 6) -> ToolResult:
    """Search the web and return titles, URLs and snippets.

    Args:
        query: What to search for.
        max_results: How many results to return.
    """
    if not query.strip():
        return ToolResult(output="empty query", ok=False)
    url = "https://html.duckduckgo.com/html/?q=" + quote_plus(query)
    try:
        response = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT
        )
    except requests.RequestException as exc:
        return ToolResult(output="search failed: %s" % exc, ok=False)
    if response.status_code >= 400:
        return ToolResult(output="search returned HTTP %d" % response.status_code, ok=False)
    results = _parse_ddg(response.text, int(max_results))
    if not results:
        return ToolResult(
            output="no results parsed for %r (the search backend may be blocked)" % query,
            ok=False,
        )
    return ToolResult(output="\n".join(results), ok=True)


def _parse_ddg(markup: str, limit: int) -> List[str]:
    rows: List[str] = []
    pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>'
        r'.*?(?:class="result__snippet"[^>]*>(?P<snippet>.*?)</a>)?',
        re.S | re.I,
    )
    for match in pattern.finditer(markup):
        href = html.unescape(match.group("href"))
        # DuckDuckGo wraps links: /l/?uddg=<encoded>
        if "uddg=" in href:
            from urllib.parse import parse_qs, unquote

            params = parse_qs(urlparse(href).query)
            href = unquote(params.get("uddg", [href])[0])
        title = html_to_text(match.group("title") or "")
        snippet = html_to_text(match.group("snippet") or "")
        rows.append("%d. %s\n   %s\n   %s" % (len(rows) + 1, title, href, snippet[:300]))
        if len(rows) >= limit:
            break
    return rows


@tool(category="web")
def open_url(url: str) -> str:
    """Open a URL in the device's default browser.

    Args:
        url: Address to open.
    """
    target = _normalise_url(url)
    from ..termux import run_termux

    ok, out = run_termux(["termux-open-url", target], timeout=15)
    if ok:
        return "opened %s" % target
    return "could not open browser (%s). URL: %s" % (out.strip(), target)
