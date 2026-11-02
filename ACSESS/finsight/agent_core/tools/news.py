"""Company news tool (Finnhub free tier).

Answers "what's been in the news about NVIDIA this week?" and supplies headlines
for the Analysis agent's sentiment work (P2) and for citations in reports.

Source: https://finnhub.io/api/v1/company-news?symbol=NVDA&from=2026-10-19&to=2026-10-26
Each article: headline, summary, source (publisher), url, datetime (Unix), related, category, id.

Rules:
  - Same key and same 60 calls/minute budget as prices.py (shared finnhub_client).
  - Free plan: North American companies; roughly the last year of news.
  - Responses are cached on disk: 15 min if the range includes today (new articles
    keep arriving), 7 days for ranges that are fully in the past.
  - Duplicates (same URL, or same headline from syndication) are removed.
  - Relevance: Finnhub tags an article to a ticker even when the company is only
    mentioned in passing (e.g. a Motley Fool piece about SpaceX that lists Nvidia
    in a disclosure). We keep an article only if its headline or summary names
    the company: "high" = in the headline, "medium" = in the summary only.
  - Articles are evidence, not advice: agents cite them with source + url + date.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone

from agent_core.tools import finnhub_client

RECENT_TTL = 15 * 60            # range includes today: refetch after 15 minutes
PAST_TTL = 7 * 24 * 60 * 60     # range fully in the past: articles won't change
MAX_DAYS_BACK = 365             # free plan covers about one year

# Names to look for in headlines/summaries (the ticker itself is always included).
# Add your companies here if they're missing; or pass names=[...] to get_company_news.
ALIASES: dict[str, list[str]] = {
    "NVDA": ["Nvidia"],
    "AAPL": ["Apple", "iPhone"],
    "MSFT": ["Microsoft"],
    "AMZN": ["Amazon", "AWS"],
    "GOOGL": ["Alphabet", "Google"],
    "GOOG": ["Alphabet", "Google"],
    "META": ["Meta Platforms", "Meta", "Facebook", "Instagram"],
    "TSLA": ["Tesla"],
    "JPM": ["JPMorgan", "JP Morgan", "J.P. Morgan"],
    "AMD": ["Advanced Micro Devices"],
    "INTC": ["Intel"],
}
RELEVANCE_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass
class NewsArticle:
    ticker: str
    headline: str
    summary: str
    source: str               # publisher, e.g. "Reuters"
    url: str
    published_at: str         # ISO UTC
    category: str = ""
    related: list[str] = field(default_factory=list)
    article_id: int | None = None
    relevance: str = ""       # "high" (named in headline), "medium" (summary), "low" (neither)
    provider: str = "Finnhub"

    @property
    def date(self) -> str:
        return self.published_at[:10]

    def to_dict(self) -> dict:
        return asdict(self)

    def citation(self) -> str:
        """Short reference for reports, e.g. 'Reuters, 2026-10-24: Nvidia ... (url)'."""
        return f"{self.source}, {self.date}: {self.headline} ({self.url})"


def _norm(headline: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", headline.lower()).strip()


def company_pattern(ticker: str, names: list[str] | None = None) -> re.Pattern:
    """Regex matching the ticker or any company name as a whole word ("Meta" but not "metaverse")."""
    terms = [ticker.upper(), *(names if names is not None else ALIASES.get(ticker.upper(), []))]
    alts = "|".join(re.escape(t) for t in sorted(set(terms), key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z0-9])\$?(?:{alts})(?![A-Za-z0-9])", re.IGNORECASE)


def relevance(headline: str, summary: str, pattern: re.Pattern) -> str:
    if pattern.search(headline):
        return "high"
    if pattern.search(summary):
        return "medium"
    return "low"


def parse_articles(ticker: str, rows: list[dict], names: list[str] | None = None,
                   min_relevance: str = "low") -> list[NewsArticle]:
    """Finnhub rows -> NewsArticle list, newest first, duplicates removed. Pure function.

    min_relevance="medium" drops articles that never name the company.
    """
    pattern = company_pattern(ticker, names)
    floor = RELEVANCE_RANK[min_relevance]
    out: list[NewsArticle] = []
    seen_urls: set[str] = set()
    seen_heads: set[str] = set()
    for r in sorted(rows or [], key=lambda r: r.get("datetime") or 0, reverse=True):
        headline = (r.get("headline") or "").strip()
        url = (r.get("url") or "").strip()
        if not headline or not url or not r.get("datetime"):
            continue
        key = _norm(headline)
        if url in seen_urls or key in seen_heads:
            continue
        summary = (r.get("summary") or "").strip()
        rel = relevance(headline, summary, pattern)
        if RELEVANCE_RANK[rel] < floor:
            continue
        seen_urls.add(url)
        seen_heads.add(key)
        out.append(NewsArticle(
            ticker=ticker.upper(),
            headline=headline,
            summary=summary,
            source=(r.get("source") or "").strip(),
            url=url,
            published_at=datetime.fromtimestamp(r["datetime"], tz=timezone.utc)
                                 .isoformat(timespec="seconds"),
            category=r.get("category") or "",
            related=[x.strip() for x in (r.get("related") or "").split(",") if x.strip()],
            article_id=r.get("id"),
            relevance=rel,
        ))
    return out


def get_company_news(ticker: str, days: int = 7, start: str | None = None,
                     end: str | None = None, limit: int | None = None,
                     min_relevance: str = "medium", names: list[str] | None = None,
                     today: date | None = None) -> list[NewsArticle]:
    """News about one company, newest first.

    get_company_news("NVDA")                       -> last 7 days, articles that name Nvidia
    get_company_news("NVDA", days=30, limit=20)    -> last 30 days, 20 newest
    get_company_news("NVDA", start="2026-08-20", end="2026-08-31")  -> around an earnings date
    min_relevance: "high" = company in the headline; "medium" (default) = headline or
                   summary; "low" = everything Finnhub tagged (incl. passing mentions).
    """
    if min_relevance not in RELEVANCE_RANK:
        raise ValueError(f"min_relevance must be one of {list(RELEVANCE_RANK)}")
    today = today or date.today()
    end_d = date.fromisoformat(end) if end else today
    start_d = date.fromisoformat(start) if start else end_d - timedelta(days=days)
    if start_d > end_d:
        raise ValueError(f"start {start_d} is after end {end_d}")
    if (today - start_d).days > MAX_DAYS_BACK:
        raise ValueError(f"Free plan covers about {MAX_DAYS_BACK} days back; {start_d} is too old")

    ttl = RECENT_TTL if end_d >= today else PAST_TTL
    rows = finnhub_client.get("/company-news",
                              {"symbol": ticker.upper(), "from": start_d.isoformat(),
                               "to": end_d.isoformat()}, ttl=ttl)
    if isinstance(rows, dict):  # Finnhub reports some problems as {"error": "..."}
        raise ValueError(f"Finnhub: {rows.get('error', rows)}")
    articles = parse_articles(ticker, rows, names=names, min_relevance=min_relevance)
    return articles[:limit] if limit else articles  # limit AFTER filtering


def get_news_for(tickers: list[str], days: int = 7, limit: int | None = 10,
                 min_relevance: str = "medium") -> dict[str, list | str]:
    """News for several companies. A failed ticker gets an error message instead of a list."""
    out: dict[str, list | str] = {}
    for t in tickers:
        try:
            out[t.upper()] = get_company_news(t, days=days, limit=limit, min_relevance=min_relevance)
        except Exception as e:  # noqa: BLE001 — one bad ticker shouldn't stop the rest
            out[t.upper()] = f"{type(e).__name__}: {e}"
    return out
