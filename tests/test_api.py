"""Tests for the HTTP API (app.main)."""

from __future__ import annotations

from starlette.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["documents_indexed"], int)


def test_ask_rejects_empty_question() -> None:
    # Validation happens before any LLM call, so this needs no model running.
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422
