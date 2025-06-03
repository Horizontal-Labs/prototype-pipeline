import pytest
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables before importing app modules
load_dotenv()

from fastapi.testclient import TestClient
from app.models import ArgumentComponent
from main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_text():
    return "Global warming is a serious threat. Because temperatures are rising worldwide, we must act now."

@pytest.fixture
def sample_components():
    return [
        ArgumentComponent(text="Global warming is a serious threat.", type="claim", confidence=0.9),
        ArgumentComponent(text="Because temperatures are rising worldwide.", type="premise", confidence=0.8)
    ] 