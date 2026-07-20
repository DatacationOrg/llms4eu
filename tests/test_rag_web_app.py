from fastapi.testclient import TestClient

from src.rag.web import app as web_app
from src.rag.web.service import ChatReply, ChatSource


def test_index_redirects_when_unauthenticated():
    client = TestClient(web_app.app)

    response = client.get("/", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"] == "/login"


def test_chat_requires_authentication():
    client = TestClient(web_app.app)

    response = client.post("/api/chat", json={"question": "hello"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required."


def test_session_reports_authenticated_user(monkeypatch):
    monkeypatch.setattr(
        web_app,
        "get_session_user",
        lambda request: {
            "name": "Ada Lovelace",
            "email": "ada@example.com",
            "username": "ada@example.com",
            "subject": "user-1",
        },
    )

    client = TestClient(web_app.app)
    response = client.get("/api/session")

    assert response.status_code == 200
    assert response.json()["authenticated"] is True
    assert response.json()["user"]["name"] == "Ada Lovelace"


def test_chat_returns_service_response(monkeypatch):
    monkeypatch.setattr(
        web_app,
        "get_session_user",
        lambda request: {
            "name": "Ada Lovelace",
            "email": "ada@example.com",
            "username": "ada@example.com",
            "subject": "user-1",
        },
    )
    monkeypatch.setattr(
        web_app.chat_service,
        "run_chat",
        lambda question, limit=None, agentic=False: ChatReply(
            question=question,
            answer="Take the lakeside forest trail.",
            agentic=agentic,
            sufficient=True,
            sources=[ChatSource(id="p1", summary="Forest trail", score=0.98)],
            attempts=[],
        ),
    )

    client = TestClient(web_app.app)
    response = client.post(
        "/api/chat",
        json={"question": "Need a quiet walk", "agentic": True, "limit": 4},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Take the lakeside forest trail."
    assert response.json()["agentic"] is True
    assert response.json()["sources"][0]["id"] == "p1"
