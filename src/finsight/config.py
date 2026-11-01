"""Project settings, read from environment variables / .env.

Every path can be overridden in .env, e.g. to keep the database in Google Drive:
    FINSIGHT_DATA_DIR="/content/drive/MyDrive/finsight_data"
"""

import os
from pathlib import Path

try:  # load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DATA_DIR = Path(os.getenv("FINSIGHT_DATA_DIR", "data"))
DB_PATH = Path(os.getenv("FINSIGHT_DB", str(DATA_DIR / "finsight.db")))
FILINGS_DIR = Path(os.getenv("FINSIGHT_FILINGS_DIR", str(DATA_DIR / "filings")))
CACHE_DIR = Path(os.getenv("FINSIGHT_CACHE_DIR", str(DATA_DIR / "cache" / "sec")))
COMPANIES_FILE = Path(os.getenv("FINSIGHT_COMPANIES", "config/companies.yaml"))
CONCEPTS_FILE = Path(os.getenv("FINSIGHT_CONCEPTS", "config/concepts.yaml"))


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def load_companies() -> list[str]:
    """The tickers FinSight supports, from config/companies.yaml."""
    return [t.upper() for t in _read_yaml(COMPANIES_FILE).get("companies", [])]


def load_concept_overrides() -> dict:
    """Per-company XBRL concept overrides, from config/concepts.yaml."""
    return {k.upper(): v for k, v in (_read_yaml(CONCEPTS_FILE).get("overrides") or {}).items()}
