import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from main import app
import json

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_analyze_valid_input():
    test_text = "Global warming is a serious threat. Since temperatures are rising worldwide, we need to act now."
    response = client.post(
        "/analyze",
        json={"text": test_text}
    )
    assert response.status_code == 200
    data = response.json()
    assert "components" in data
    assert "relations" in data
    assert isinstance(data["components"], list)
    assert isinstance(data["relations"], list)
    assert all(["text" in comp and "type" in comp and "confidence" in comp 
                for comp in data["components"]])

def test_analyze_empty_input():
    response = client.post(
        "/analyze",
        json={"text": ""}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["components"] == []
    assert data["relations"] == []

def test_analyze_invalid_input():
    response = client.post(
        "/analyze",
        json={"invalid_field": "test"}
    )
    assert response.status_code == 422  # Validation error

def test_analyze_long_text():
    long_text = " ".join(["This is a test sentence."] * 100)
    response = client.post(
        "/analyze",
        json={"text": long_text}
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["components"]) > 0 