"""
Tests for the /ask endpoint.
OpenAI and Qdrant calls are mocked so no real credentials are needed.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")


@pytest.fixture
def client():
    fake_chain_result = {
        "resposta": "O YAITEC Atende é uma plataforma de atendimento ao cliente com IA. [Fonte: sobre-o-yaitec-atende.md]",
        "fontes": ["sobre-o-yaitec-atende.md"],
    }

    fake_chain = MagicMock(return_value=fake_chain_result)

    with patch("app.retrieval_pipeline.get_vector_store", return_value=MagicMock()), \
         patch("app.rag.build_rag_chain", return_value=fake_chain), \
         patch("app.main._rag_chain", fake_chain):
        from app.main import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_resposta_and_fontes(client):
    response = client.post("/ask", json={"pergunta": "O que é o YAITEC Atende?"})
    assert response.status_code == 200
    data = response.json()
    assert "resposta" in data
    assert "fontes" in data
    assert isinstance(data["resposta"], str)
    assert isinstance(data["fontes"], list)
    assert len(data["resposta"]) > 0


def test_ask_empty_question(client):
    response = client.post("/ask", json={"pergunta": "   "})
    assert response.status_code == 422


def test_ask_missing_field(client):
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_root_serves_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_ask_audio_returns_transcription_answer_and_audio(client):
    with patch("app.main.transcribe", return_value="Quais são os planos?") as mock_stt, \
         patch("app.main.synthesize", return_value=b"fake-mp3-bytes") as mock_tts:
        response = client.post(
            "/ask-audio",
            files={"file": ("audio.webm", b"fake-audio-bytes", "audio/webm")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["pergunta"] == "Quais são os planos?"
    assert isinstance(data["resposta"], str) and len(data["resposta"]) > 0
    assert isinstance(data["fontes"], list)
    assert isinstance(data["audio_base64"], str) and len(data["audio_base64"]) > 0
    mock_stt.assert_called_once()
    mock_tts.assert_called_once()


def test_ask_audio_empty_file(client):
    response = client.post(
        "/ask-audio",
        files={"file": ("audio.webm", b"", "audio/webm")},
    )
    assert response.status_code == 422
