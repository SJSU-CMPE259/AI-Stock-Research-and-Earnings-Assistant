"""Earnings-call transcripts tool (Alpha Vantage, free key).

Answers "what did NVIDIA's CEO say about demand on the last earnings call?" by giving
the RAG index the call's text, speaker by speaker.

Source (one call = one company, one quarter):
  https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol=NVDA&quarter=2027Q2&apikey=...
Response:
  {"symbol": "NVDA", "quarter": "2027Q2",
   "transcript": [{"speaker": "Jensen Huang", "title": "President and CEO",
                   "content": "...", "sentiment": "0.6"}, ...]}

Why Alpha Vantage (checked Nov 2026): FMP puts transcripts in its paid Ultimate plan,
Finnhub and API Ninjas mark them premium, and scraping Motley Fool is against its terms.
Alpha Vantage lists EARNINGS_CALL_TRANSCRIPT without a premium tag, back to 2010Q1.

Rules:
  - Key: ALPHAVANTAGE_API_KEY in .env (free from https://www.alphavantage.co/support/#api-key).
    Sent as a URL parameter (Alpha Vantage requires that), so never log full request URLs.
  - Free key: 25 requests per day. Alpha Vantage offers unlimited requests to verified
    educational projects; ask them if the class project needs more.
  - `quarter` is "YYYYQn". Alpha Vantage documents it as the FISCAL quarter, which would make
    NVIDIA's Q2 FY2027 "2027Q2". Confirm once per company with
    `scripts/ingest_transcripts.py --check`, which prints the quarter the call itself names.
  - Limits and errors come back as HTTP 200 with {"Information": ...}, {"Note": ...} or
    {"Error Message": ...}, so the body is checked, not just the status code.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

BASE_URL = "https://www.alphavantage.co/query"
MIN_INTERVAL = 1.2           # seconds between calls
SOURCE = "Alpha Vantage"

_session = requests.Session()
_last_call = 0.0


class TranscriptAccessError(RuntimeError):
    """Bad or missing key, or the endpoint is premium for this key."""


class TranscriptLimitError(RuntimeError):
    """Daily or per-minute request limit reached: try again later (free key: 25/day)."""


@dataclass
class Turn:
    speaker: str
    title: str
    content: str
    sentiment: float | None = None   # Alpha Vantage's per-turn score, -1..1

    @property
    def n_words(self) -> int:
        return len(self.content.split())


@dataclass
class Transcript:
    ticker: str
    quarter: str                 # "2027Q2"
    turns: list[Turn] = field(default_factory=list)
    source: str = SOURCE

    @property
    def available(self) -> bool:
        return bool(self.turns)

    @property
    def n_words(self) -> int:
        return sum(t.n_words for t in self.turns)

    @property
    def source_url(self) -> str:
        """Where it came from, WITHOUT the API key (safe to store and cite)."""
        return (f"{BASE_URL}?function=EARNINGS_CALL_TRANSCRIPT&symbol={self.ticker}"
                f"&quarter={self.quarter}")

    def to_dict(self) -> dict:
        return {"ticker": self.ticker, "quarter": self.quarter, "source": self.source,
                "source_url": self.source_url, "turns": [asdict(t) for t in self.turns]}

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        return cls(ticker=d["ticker"], quarter=d["quarter"], source=d.get("source", SOURCE),
                   turns=[Turn(**t) for t in d.get("turns", [])])


def quarter_key(fiscal_year: int, fiscal_period: str) -> str:
    """(2027, "Q2") -> "2027Q2"; a 10-K's "FY" is the fourth quarter's call -> "2026Q4"."""
    q = "Q4" if fiscal_period == "FY" else fiscal_period
    if q not in ("Q1", "Q2", "Q3", "Q4"):
        raise ValueError(f"not a fiscal quarter: {fiscal_period!r}")
    return f"{int(fiscal_year)}{q}"


def api_key() -> str:
    key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    if not key:
        raise TranscriptAccessError(
            "Set ALPHAVANTAGE_API_KEY in your .env (free from https://www.alphavantage.co/support/#api-key)")
    return key


def _float(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_transcript(ticker: str, quarter: str, data: dict) -> Transcript:
    """Alpha Vantage JSON -> Transcript. Pure function (unit-tested). Raises on limit/error bodies."""
    if not isinstance(data, dict):
        raise ValueError(f"unexpected response: {str(data)[:200]}")
    msg = data.get("Information") or data.get("Note") or data.get("Error Message")
    if msg and "transcript" not in data:
        low = msg.lower()
        # check limits first: the daily-limit message also mentions "API key" and "premium plans"
        if "rate limit" in low or "per day" in low or "requests per" in low or "frequency" in low:
            raise TranscriptLimitError(f"Alpha Vantage: {msg}")
        raise TranscriptAccessError(f"Alpha Vantage: {msg}")   # premium-only, bad key, bad call
    turns = []
    for row in data.get("transcript") or []:
        content = (row.get("content") or "").strip()
        if not content:
            continue
        turns.append(Turn(speaker=(row.get("speaker") or "").strip(),
                          title=(row.get("title") or "").strip(),
                          content=content, sentiment=_float(row.get("sentiment"))))
    return Transcript(ticker=ticker.upper(), quarter=data.get("quarter") or quarter, turns=turns)


def get_transcript(ticker: str, quarter: str) -> Transcript:
    """One quarter's call, e.g. get_transcript("NVDA", "2027Q2"). Empty `turns` = not published."""
    global _last_call
    wait = MIN_INTERVAL - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    params = {"function": "EARNINGS_CALL_TRANSCRIPT", "symbol": ticker.upper(),
              "quarter": quarter, "apikey": api_key()}
    for attempt in range(3):
        _last_call = time.time()
        resp = _session.get(BASE_URL, params=params, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(2 ** attempt * 2)
            continue
        if resp.status_code >= 400:   # don't use raise_for_status: its message includes the URL + key
            raise TranscriptAccessError(f"Alpha Vantage HTTP {resp.status_code}")
        return parse_transcript(ticker, quarter, resp.json())
    raise TranscriptLimitError(f"Alpha Vantage HTTP {resp.status_code} after 3 tries")
