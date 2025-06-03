import os
import pytest
from unittest.mock import patch

@pytest.fixture(autouse=True)
def mock_env_vars():
    """
    Fixture to ensure environment variables are available during testing.
    If HF_TOKEN is not in environment, use a mock token.
    """
    with patch.dict(os.environ, {
        "HF_TOKEN": os.getenv("HF_TOKEN", "mock_token_for_testing")
    }):
        yield 