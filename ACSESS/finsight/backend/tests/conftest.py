"""Shared pytest fixtures for FinSight backend tests."""
import pytest


@pytest.fixture
def mock_sec_user_agent():
    """Mock SEC User-Agent string."""
    return "Test User test@example.com"


# TODO: Add fixtures for:
# - in-memory SQLite database
# - mock SEC API responses
# - mock embeddings
# - fake LLM client
