"""Shared HTTP client for Finnhub (used by prices.py and news.py).

Like sec_client.py for the SEC: one place for the key, the rate limit and the cache.
All Finnhub tools share ONE free-tier budget (about 60 calls/minute per key), so
they must share one rate limiter; two separate limiters could exceed it together.

  - Key: FINNHUB_API_KEY in .env (free from https://finnhub.io). Sent in the
    X-Finnhub-Token header, so it never appears in URLs or logs.
  - Rate limit: >= 1.1 s between calls.
  - Optional disk cache (data/cache/finnhub/) for responses that don't change much.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load .env from the finsight repo root (this file is <root>/agent_core/tools/finnhub_client.py)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

BASE_URL = "https://finnhub.io/api/v1"
CACHE_DIR = Path(os.getenv("FINSIGHT_CACHE_DIR", "data/cache/sec")).parent / "finnhub"
MIN_INTERVAL = 1.1  # seconds between calls -> under 60 calls/minute

_session = requests.Session()
_last_call = 0.0


def api_key() -> str:
    key = os.getenv("FINNHUB_API_KEY", "")
    if not key:
        raise RuntimeError("Set FINNHUB_API_KEY in your .env (free key from https://finnhub.io)")
    return key


def _cache_path(path: str, params: dict) -> Path:
    key = hashlib.sha256(f"{path}?{sorted(params.items())}".encode()).hexdigest()[:24]
    return CACHE_DIR / f"{key}.json"


def get(path: str, params: dict, ttl: int = 0):
    """GET https://finnhub.io/api/v1{path}. ttl > 0 reuses a saved response for ttl seconds."""
    cache_file = _cache_path(path, params) if ttl > 0 else None
    if cache_file and cache_file.exists() and time.time() - cache_file.stat().st_mtime < ttl:
        return json.loads(cache_file.read_text())

    global _last_call
    for attempt in range(3):
        wait = MIN_INTERVAL - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        resp = _session.get(BASE_URL + path, params=params,
                            headers={"X-Finnhub-Token": api_key()}, timeout=20)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt * 2)  # 2s, 4s, 8s
            continue
        if resp.status_code in (401, 403):
            raise RuntimeError("Finnhub rejected the request: check FINNHUB_API_KEY, "
                               "or this endpoint/date range isn't on the free plan")
        resp.raise_for_status()
        data = resp.json()
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data))
        return data
    resp.raise_for_status()
    return resp.json()
