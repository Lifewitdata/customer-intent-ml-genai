"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_baseline(client):
    r = client.post("/predict", json={"text": "my card is expiring", "model": "baseline"})
    assert r.status_code == 200
    data = r.json()
    assert "intent" in data
    assert "confidence" in data
    assert data["model"] == "baseline"


def test_rag(client):
    r = client.post("/rag", json={"text": "my card is expiring", "k": 2})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert len(data["results"]) == 2


def test_agent(client):
    r = client.post("/agent", json={"text": "my card is expiring"})
    assert r.status_code == 200
    data = r.json()
    assert data["action"] in {"answer", "escalate", "clarify"}
    assert "response" in data
