import json
import pytest

from src.finsight.tools import sec_client


def test_missing_user_agent(monkeypatch):
    monkeypatch.setattr(sec_client, "USER_AGENT", "")

    with pytest.raises(RuntimeError):
        sec_client._headers()


def test_valid_user_agent(monkeypatch):
    monkeypatch.setattr(
        sec_client,
        "USER_AGENT",
        "FinSight Team1 test@example.com"
    )

    headers = sec_client._headers()

    assert headers["User-Agent"] == "FinSight Team1 test@example.com"


def test_json_caching(monkeypatch, tmp_path):
    monkeypatch.setattr(sec_client, "CACHE_DIR", tmp_path)

    calls = []

    class FakeResponse:
        def json(self):
            return {"name": "Apple Inc.", "cik": 320193}

    def fake_fetch(url):
        calls.append(url)
        return FakeResponse()

    monkeypatch.setattr(sec_client, "_fetch", fake_fetch)

    url = "https://example.com/company.json"

    first = sec_client.get_json(url)
    second = sec_client.get_json(url)

    assert first == second
    assert first["name"] == "Apple Inc."
    assert len(calls) == 1
    assert sec_client._cache_path(url, ".json").exists()
