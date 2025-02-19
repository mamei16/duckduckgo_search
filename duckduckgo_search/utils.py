from __future__ import annotations

import gzip
import re
import ssl
from html import unescape
from pathlib import Path
from random import SystemRandom, choice, choices, randint
from typing import Any
from urllib.parse import unquote

import certifi
import h2
import httpcore

from .exceptions import DuckDuckGoSearchException

try:
    HAS_ORJSON = True
    import orjson
except ImportError:
    HAS_ORJSON = False
    import json

REGEX_STRIP_TAGS = re.compile("<.*?>")


def json_dumps(obj: Any) -> str:
    try:
        return (
            orjson.dumps(obj, option=orjson.OPT_INDENT_2).decode()
            if HAS_ORJSON
            else json.dumps(obj, ensure_ascii=False, indent=2)
        )
    except Exception as ex:
        raise DuckDuckGoSearchException(f"{type(ex).__name__}: {ex}") from ex


def json_loads(obj: str | bytes) -> Any:
    try:
        return orjson.loads(obj) if HAS_ORJSON else json.loads(obj)
    except Exception as ex:
        raise DuckDuckGoSearchException(f"{type(ex).__name__}: {ex}") from ex


def _extract_vqd(html_bytes: bytes, keywords: str) -> str:
    """Extract vqd from html bytes."""
    for c1, c1_len, c2 in (
        (b'vqd="', 5, b'"'),
        (b"vqd=", 4, b"&"),
        (b"vqd='", 5, b"'"),
    ):
        try:
            start = html_bytes.index(c1) + c1_len
            end = html_bytes.index(c2, start)
            return html_bytes[start:end].decode()
        except ValueError:
            pass
    raise DuckDuckGoSearchException(f"_extract_vqd() {keywords=} Could not extract vqd.")


def _normalize(raw_html: str) -> str:
    """Strip HTML tags from the raw_html string."""
    return unescape(REGEX_STRIP_TAGS.sub("", raw_html)) if raw_html else ""


def _normalize_url(url: str) -> str:
    """Unquote URL and replace spaces with '+'."""
    return unquote(url).replace(" ", "+") if url else ""


def _expand_proxy_tb_alias(proxy: str | None) -> str | None:
    """Expand "tb" to a full proxy URL if applicable."""
    return "socks5://127.0.0.1:9150" if proxy == "tb" else proxy


# SSL
CRYPTORAND = SystemRandom()
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
DEFAULT_CIPHERS = [  # https://developers.cloudflare.com/ssl/reference/cipher-suites/recommendations/
    # Modern:
    "ECDHE-ECDSA-AES128-GCM-SHA256", "ECDHE-ECDSA-CHACHA20-POLY1305", "ECDHE-RSA-AES128-GCM-SHA256",
    "ECDHE-RSA-CHACHA20-POLY1305", "ECDHE-ECDSA-AES256-GCM-SHA384", "ECDHE-RSA-AES256-GCM-SHA384",
    # Compatible:
    "ECDHE-ECDSA-AES128-GCM-SHA256", "ECDHE-ECDSA-CHACHA20-POLY1305", "ECDHE-RSA-AES128-GCM-SHA256",
    "ECDHE-RSA-CHACHA20-POLY1305", "ECDHE-ECDSA-AES256-GCM-SHA384", "ECDHE-RSA-AES256-GCM-SHA384",
    "ECDHE-ECDSA-AES128-SHA256", "ECDHE-RSA-AES128-SHA256", "ECDHE-ECDSA-AES256-SHA384",  "ECDHE-RSA-AES256-SHA384",
    # Legacy:
    "ECDHE-ECDSA-AES128-SHA", "ECDHE-RSA-AES128-SHA", "AES128-GCM-SHA256", "AES128-SHA256", "AES128-SHA",
    "ECDHE-RSA-AES256-SHA", "AES256-GCM-SHA384", "AES256-SHA256", "AES256-SHA", "DES-CBC3-SHA",
]  # fmt: skip


def _get_random_ssl_context() -> ssl.SSLContext:
    """Get SSL context with shuffled ciphers."""
    shuffled_ciphers = CRYPTORAND.sample(DEFAULT_CIPHERS[6:], len(DEFAULT_CIPHERS) - 6)
    SSL_CONTEXT.set_ciphers(":".join(DEFAULT_CIPHERS[:6] + shuffled_ciphers))
    return SSL_CONTEXT


# Headers
script_dir = Path(__file__).parent
file_path = script_dir / "headers.json.gz"
with gzip.open(file_path, "rt") as f:
    DEFAULT_HEADERS = json_loads(f.read())
HEADERS: list[dict[str, str]] = [item["header"] for item in DEFAULT_HEADERS if isinstance(item["header"], dict)]
HEADERS_PROB: list[float] = [item["probability"] for item in DEFAULT_HEADERS if isinstance(item["probability"], float)]


def _get_probability_headers() -> dict[str, str]:
    """Get probability headers using probability."""
    return choices(HEADERS, weights=HEADERS_PROB)[0]


def _get_random_headers() -> dict[str, str]:
    """Get random headers."""
    return choice(HEADERS)


# Patch httpcore
def _send_connection_init(self, request):  # type: ignore
    self._h2_state.local_settings = h2.settings.Settings(
        client=True,
        initial_values={
            h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: randint(100, 200),
            h2.settings.SettingCodes.HEADER_TABLE_SIZE: randint(4000, 5000),
            h2.settings.SettingCodes.MAX_FRAME_SIZE: randint(16384, 17384),
            h2.settings.SettingCodes.MAX_CONCURRENT_STREAMS: randint(100, 200),
            h2.settings.SettingCodes.MAX_HEADER_LIST_SIZE: randint(65500, 66500),
            h2.settings.SettingCodes.ENABLE_CONNECT_PROTOCOL: randint(0, 1),
            h2.settings.SettingCodes.ENABLE_PUSH: randint(0, 1),
        },
    )
    self._h2_state.initiate_connection()
    self._h2_state.increment_flow_control_window(2**24)
    self._write_outgoing_data(request)


httpcore._sync.http2.HTTP2Connection._send_connection_init = _send_connection_init  # type: ignore
