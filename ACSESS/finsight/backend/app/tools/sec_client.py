"""SEC HTTP client with throttling, retry, and disk caching."""
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# SEC requires max 10 req/s; we use 8 to be safe
THROTTLE_DELAY = 1.0 / 8  # ~125ms between requests
CACHE_DIR = settings.data_dir / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class SECClient:
    """Throttled SEC API client with caching and retry logic."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": settings.sec_user_agent, "Accept-Encoding": "gzip"}
        )

        # Retry strategy: exponential backoff on 429 and 5xx
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.last_request_time = 0

    def _get_cache_path(self, url: str) -> Path:
        """Generate a deterministic cache path for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return CACHE_DIR / f"{url_hash}.json"

    def _read_cache(self, url: str) -> Optional[dict]:
        """Read from disk cache if it exists."""
        cache_path = self._get_cache_path(url)
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return None

    def _write_cache(self, url: str, data: dict) -> None:
        """Write response to disk cache."""
        cache_path = self._get_cache_path(url)
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except IOError as e:
            logger.warning(f"Failed to cache response for {url}: {e}")

    def get(self, url: str, use_cache: bool = True) -> dict:
        """Fetch URL with throttling, retry, and caching.

        Args:
            url: Full URL to fetch
            use_cache: Whether to use disk cache

        Returns:
            Parsed JSON response
        """
        # Check cache first
        if use_cache:
            cached = self._read_cache(url)
            if cached is not None:
                logger.debug(f"Cache hit: {url}")
                return cached

        # Throttle to max 8 req/s
        elapsed = time.time() - self.last_request_time
        if elapsed < THROTTLE_DELAY:
            time.sleep(THROTTLE_DELAY - elapsed)
        self.last_request_time = time.time()

        # Make request
        logger.debug(f"GET {url}")
        response = self.session.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Cache successful response
        if use_cache:
            self._write_cache(url, data)

        return data


def get_sec_client() -> SECClient:
    """Get or create the SEC client singleton."""
    if not hasattr(get_sec_client, "_client"):
        get_sec_client._client = SECClient()
    return get_sec_client._client
