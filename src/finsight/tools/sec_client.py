"""Shared HTTP client for SEC EDGAR.

The SEC requires two things from every caller:
  1. A User-Agent header that identifies you, e.g. "FinSight Team1 you@sjsu.edu".
     Requests without it get HTTP 403.
  2. No more than 10 requests per second. We stay well under that.

Responses are cached on disk so repeated runs (and teammates' notebooks)
don't hit the SEC again for the same data.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

USER_AGENT = os.getenv("SEC_USER_AGENT", "")
CACHE_DIR = Path(os.getenv("FINSIGHT_CACHE_DIR", "data/cache/sec"))
CACHE_TTL_SECONDS = 24 * 60 * 60  # refetch after a day so new filings show up
MIN_INTERVAL = 0.15  # seconds between requests (about 6-7 per second)

_last_request = 0.0
_session = requests.Session()


def _headers() -> dict:
    if not USER_AGENT:
        raise RuntimeError(
            "Set SEC_USER_AGENT in your .env, e.g. 'FinSight Team1 you@sjsu.edu'"
        )
    return {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def _cache_path(url: str, suffix: str) -> Path:
    key = hashlib.sha256(url.encode()).hexdigest()[:24]
    return CACHE_DIR / f"{key}{suffix}"


def _fetch(url: str) -> requests.Response:
    global _last_request
    wait = MIN_INTERVAL - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    for attempt in range(3):
        _last_request = time.time()
        resp = _session.get(url, headers=_headers(), timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt)  # back off and retry
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def _fresh(path: Path, ttl: int) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl


def get_json(url: str, ttl: int = CACHE_TTL_SECONDS) -> dict:
    path = _cache_path(url, ".json")
    if _fresh(path, ttl):
        return json.loads(path.read_text())
    data = _fetch(url).json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def get_text(url: str, ttl: int = 365 * 24 * 60 * 60) -> str:
    """Filing documents never change once filed, so cache them for a long time."""
    path = _cache_path(url, ".html")
    if _fresh(path, ttl):
        return path.read_text(errors="replace")
    text = _fetch(url).text
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text
