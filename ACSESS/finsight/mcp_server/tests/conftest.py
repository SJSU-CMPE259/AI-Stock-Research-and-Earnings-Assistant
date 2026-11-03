"""Shared test setup. Offline only: Finnhub and SEC are stubbed in each test file."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # ACSESS/finsight
sys.path.insert(0, str(ROOT))

# Set before any mcp_server/backend import: fake mode (FakeLLM, keyword routing)
# and a dummy SEC User-Agent so backend/app/settings.py validates.
os.environ["FINSIGHT_FAKE"] = "1"
os.environ.setdefault("SEC_USER_AGENT", "FinSight Test test@example.com")


def pytest_configure(config):
    config.addinivalue_line("markers", "network: needs real Finnhub/SEC access (run with -m network)")
