-- FinSight local database (SQLite). Rebuilt from the SEC by scripts/build_db.py.

CREATE TABLE IF NOT EXISTS companies (
    ticker          TEXT PRIMARY KEY,          -- "NVDA"
    cik             TEXT NOT NULL,             -- "0001045810"
    name            TEXT,                      -- "NVIDIA CORP"
    fiscal_year_end TEXT,                      -- "0125" = fiscal year ends around Jan 25
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS filings (
    accession       TEXT PRIMARY KEY,          -- unique filing ID, used in citations
    ticker          TEXT NOT NULL REFERENCES companies(ticker),
    form            TEXT NOT NULL,             -- 10-K, 10-Q, 8-K
    filing_date     TEXT NOT NULL,
    period_end      TEXT,                      -- period covered (10-K/10-Q) or event date (8-K)
    fiscal_year     INTEGER,                   -- from the filing, e.g. 2027
    fiscal_period   TEXT,                      -- Q1, Q2, Q3, FY (10-K), Q1..Q4 for earnings 8-Ks
    items           TEXT,                      -- 8-K items, e.g. "2.02,9.01"
    document_url    TEXT,                      -- main document on sec.gov
    local_path      TEXT,                      -- downloaded copy, if any
    added_at        TEXT NOT NULL              -- when we first stored it (for new-filing checks)
);
CREATE INDEX IF NOT EXISTS idx_filings_ticker_form ON filings(ticker, form, filing_date);

CREATE TABLE IF NOT EXISTS financial_facts (
    id              INTEGER PRIMARY KEY,
    ticker          TEXT NOT NULL REFERENCES companies(ticker),
    metric          TEXT NOT NULL,             -- revenue, net_income, eps_diluted, ...
    concept         TEXT NOT NULL,             -- XBRL concept actually used
    unit            TEXT NOT NULL,             -- USD, USD/shares
    period_type     TEXT NOT NULL,             -- quarter, annual, instant
    period_start    TEXT,                      -- NULL for instant (balance-sheet) values
    period_end      TEXT NOT NULL,
    fiscal_year     INTEGER NOT NULL,
    fiscal_period   TEXT NOT NULL,             -- Q1..Q4, FY
    value           REAL NOT NULL,
    form            TEXT NOT NULL,
    accession       TEXT NOT NULL,             -- filing the value came from
    filed           TEXT NOT NULL,
    derived         INTEGER NOT NULL DEFAULT 0, -- 1 = computed (Q4 = FY - 9 months)
    source_url      TEXT NOT NULL,
    UNIQUE (ticker, metric, period_type, period_end)  -- no duplicate periods
);
CREATE INDEX IF NOT EXISTS idx_facts_lookup ON financial_facts(ticker, metric, period_type, period_end);

CREATE TABLE IF NOT EXISTS ingest_log (
    id              INTEGER PRIMARY KEY,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    ticker          TEXT,
    new_filings     INTEGER DEFAULT 0,
    new_facts       INTEGER DEFAULT 0,
    status          TEXT,                      -- ok / error
    error           TEXT
);
