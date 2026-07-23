from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from src.rag.web import service as chat_service
from src.shared.env import load_local_env, load_yaml


load_local_env()

CONFIG = load_yaml(Path(__file__).resolve().parents[1] / "config.yaml")
oauth = OAuth()


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Tourism RAG Chat</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    :root {
      --bg: #f5f1e8;
      --panel: rgba(255, 252, 245, 0.88);
      --ink: #1f2933;
      --muted: #5f6c7b;
      --accent: #0f766e;
      --accent-strong: #115e59;
      --line: rgba(31, 41, 51, 0.12);
      --warm: #d97706;
      --shadow: 0 24px 60px rgba(31, 41, 51, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: 'Space Grotesk', sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(217, 119, 6, 0.22), transparent 34%),
        radial-gradient(circle at right, rgba(15, 118, 110, 0.18), transparent 28%),
        linear-gradient(180deg, #fbf7ef 0%, var(--bg) 100%);
    }
    main {
      width: min(1100px, calc(100vw - 32px));
      margin: 40px auto;
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 20px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }
    .sidebar {
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .eyebrow {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--warm);
    }
    h1 {
      font-size: clamp(2rem, 4vw, 3rem);
      line-height: 0.98;
      margin: 0;
    }
    p {
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }
    .user-card,
    .source-card,
    .attempt-card,
    .answer-card {
      border: 1px solid var(--line);
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.7);
    }
    .user-card,
    .source-card,
    .attempt-card {
      padding: 16px;
    }
    .workspace {
      padding: 24px;
      display: grid;
      gap: 18px;
    }
    .composer {
      display: grid;
      gap: 12px;
    }
    textarea {
      width: 100%;
      min-height: 150px;
      resize: vertical;
      border-radius: 22px;
      border: 1px solid var(--line);
      padding: 18px;
      font: inherit;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.84);
    }
    textarea:focus,
    input:focus {
      outline: 2px solid rgba(15, 118, 110, 0.25);
      border-color: rgba(15, 118, 110, 0.45);
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }
    .button {
      border: none;
      border-radius: 999px;
      padding: 14px 20px;
      font: inherit;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease, background 120ms ease;
    }
    .button:hover { transform: translateY(-1px); }
    .button:disabled { opacity: 0.6; cursor: wait; transform: none; }
    .button-primary { background: var(--accent); color: white; }
    .button-primary:hover { background: var(--accent-strong); }
    .button-secondary {
      background: rgba(31, 41, 51, 0.06);
      color: var(--ink);
      border: 1px solid var(--line);
    }
    .inline-field {
      display: inline-flex;
      gap: 10px;
      align-items: center;
      padding: 10px 14px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.72);
      color: var(--muted);
    }
    .inline-field input { width: 68px; border: none; background: transparent; font: inherit; color: var(--ink); }
    .stack {
      display: grid;
      gap: 12px;
    }
    .answer-card {
      padding: 22px;
      display: grid;
      gap: 14px;
      min-height: 220px;
    }
    .answer-text { white-space: pre-wrap; line-height: 1.6; }
    .meta { color: var(--muted); font-size: 14px; }
    .hidden { display: none !important; }
    .list {
      display: grid;
      gap: 12px;
    }
    .source-title,
    .attempt-title {
      font-family: 'IBM Plex Mono', monospace;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 8px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(15, 118, 110, 0.1);
      color: var(--accent-strong);
      width: fit-content;
    }
    .error {
      color: #b42318;
      background: rgba(180, 35, 24, 0.08);
      border: 1px solid rgba(180, 35, 24, 0.18);
      border-radius: 16px;
      padding: 14px 16px;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      .sidebar { order: 2; }
    }
  </style>
</head>
<body>
  <main>
    <section class="panel sidebar">
      <div>
        <div class="eyebrow">Authenticated local RAG</div>
        <h1>Ask the tourism agent.</h1>
      </div>
      <p>Single-turn chat on top of the repo's retrieval pipeline, gated with your Microsoft Entra sign-in.</p>
      <div id="user-card" class="user-card hidden"></div>
      <div class="user-card">
        <div class="source-title">How it works</div>
        <p>Questions are answered from locally indexed tourism data. Toggle agentic retrieval when you want the system to retry for better context.</p>
      </div>
      <button id="logout" class="button button-secondary">Log out</button>
    </section>

    <section class="panel workspace">
      <div class="composer">
        <textarea id="question" placeholder="What place is best for a quiet forest walk near water?"></textarea>
        <div class="controls">
          <label class="inline-field"><input id="agentic" type="checkbox" />Agentic retrieval</label>
          <label class="inline-field">Limit <input id="limit" type="number" min="1" max="20" value="3" /></label>
          <button id="submit" class="button button-primary">Ask</button>
        </div>
      </div>

      <div id="error" class="error hidden"></div>

      <section class="answer-card">
        <div class="eyebrow">Answer</div>
        <div id="status" class="badge">Ready</div>
        <div id="answer" class="answer-text">Your answer will appear here.</div>
        <div id="meta" class="meta"></div>
      </section>

      <section class="stack">
        <div>
          <div class="source-title">Retrieved sources</div>
          <div id="sources" class="list"></div>
        </div>
        <div>
          <div class="attempt-title">Agentic attempts</div>
          <div id="attempts" class="list"></div>
        </div>
      </section>
    </section>
  </main>

  <script>
    const userCard = document.getElementById('user-card');
    const question = document.getElementById('question');
    const agentic = document.getElementById('agentic');
    const limit = document.getElementById('limit');
    const submit = document.getElementById('submit');
    const logout = document.getElementById('logout');
    const answer = document.getElementById('answer');
    const meta = document.getElementById('meta');
    const status = document.getElementById('status');
    const error = document.getElementById('error');
    const sources = document.getElementById('sources');
    const attempts = document.getElementById('attempts');

    async function loadSession() {
      const response = await fetch('/api/session');
      const data = await response.json();
      if (!data.authenticated || !data.user) {
        window.location.href = '/login';
        return;
      }
      userCard.classList.remove('hidden');
      userCard.innerHTML = [
        '<div class="source-title">Signed in</div>',
        `<strong>${data.user.name}</strong>`,
        `<p>${data.user.email || data.user.username || ''}</p>`
      ].join('');
    }

    function setLoading(isLoading) {
      submit.disabled = isLoading;
      status.textContent = isLoading ? 'Thinking...' : 'Ready';
    }

    function setError(message) {
      if (!message) {
        error.classList.add('hidden');
        error.textContent = '';
        return;
      }
      error.classList.remove('hidden');
      error.textContent = message;
    }

    function renderSources(items) {
      sources.innerHTML = '';
      if (!items.length) {
        sources.innerHTML = '<div class="source-card"><p>No retrieved sources were returned.</p></div>';
        return;
      }
      items.forEach((item) => {
        const element = document.createElement('div');
        element.className = 'source-card';
        element.innerHTML = `
          <div class="source-title">${item.id}</div>
          <strong>${item.summary}</strong>
          <p>Similarity score: ${item.score.toFixed(3)}</p>
        `;
        sources.appendChild(element);
      });
    }

    function renderAttempts(items) {
      attempts.innerHTML = '';
      if (!items.length) {
        attempts.innerHTML = '<div class="attempt-card"><p>Agentic retrieval was not used for this answer.</p></div>';
        return;
      }
      items.forEach((item) => {
        const element = document.createElement('div');
        element.className = 'attempt-card';
        element.innerHTML = `
          <div class="attempt-title">Attempt ${item.attempt}</div>
          <strong>${item.strategy}</strong>
          <p>model=${item.embedding_model} limit=${item.limit} hits=${item.hits}</p>
          <p>${item.reason}</p>
        `;
        attempts.appendChild(element);
      });
    }

    async function ask() {
      const text = question.value.trim();
      if (!text) {
        setError('Enter a question first.');
        return;
      }
      setError('');
      setLoading(true);
      try {
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: text,
            agentic: agentic.checked,
            limit: Number(limit.value) || 3,
          }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || 'Chat request failed.');
        }
        answer.textContent = data.answer;
        meta.textContent = data.sufficient === null
          ? `Returned ${data.sources.length} retrieved source(s).`
          : `Sufficient context: ${data.sufficient ? 'yes' : 'no'}. Returned ${data.sources.length} source(s).`;
        renderSources(data.sources || []);
        renderAttempts(data.attempts || []);
      } catch (err) {
        setError(err.message || 'Chat request failed.');
      } finally {
        setLoading(false);
      }
    }

    submit.addEventListener('click', ask);
    logout.addEventListener('click', async () => {
      await fetch('/logout', { method: 'POST' });
      window.location.href = '/login';
    });

    loadSession();
  </script>
</body>
</html>"""


class UserSession(BaseModel):
    name: str
    email: str | None = None
    username: str | None = None
    subject: str


class SessionResponse(BaseModel):
    authenticated: bool
    auth_configured: bool
    user: UserSession | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    limit: int | None = Field(default=None, ge=1, le=20)
    agentic: bool = False


@dataclass(frozen=True)
class EntraSettings:
    tenant_id: str
    client_id: str
    client_secret: str
    redirect_uri: str | None

    @property
    def metadata_url(self) -> str:
        return (
            "https://login.microsoftonline.com/"
            f"{self.tenant_id}/v2.0/.well-known/openid-configuration"
        )


def create_app() -> FastAPI:
    app = FastAPI(title="Tourism RAG Chat")
    app.add_middleware(
        SessionMiddleware,
        secret_key=os.getenv("CHAT_SESSION_SECRET", "local-chat-session-secret"),
        same_site="lax",
        session_cookie=CONFIG.get("web_session_cookie", "tourism_rag_session"),
    )

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Response:
        if get_session_user(request) is None:
            return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
        return HTMLResponse(HTML)

    @app.get("/login")
    async def login(request: Request):
        if get_session_user(request) is not None:
            return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)
        client = get_entra_client()
        redirect_uri = _redirect_uri(request)
        return await client.authorize_redirect(request, redirect_uri)

    @app.get("/auth/callback")
    async def auth_callback(request: Request):
        client = get_entra_client()
        try:
            token = await client.authorize_access_token(request)
        except OAuthError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Login failed: {exc.error}",
            ) from exc
        claims = token.get("userinfo") or await client.parse_id_token(request, token)
        if not claims:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Missing user claims from Entra callback.",
            )
        request.session["user"] = _claims_to_session_user(claims)
        return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)

    @app.post("/logout")
    async def logout(request: Request) -> dict[str, bool]:
        request.session.clear()
        return {"ok": True}

    @app.get("/api/session", response_model=SessionResponse)
    async def session(request: Request) -> SessionResponse:
        user = get_session_user(request)
        if user is None:
            return SessionResponse(
                authenticated=False, auth_configured=is_auth_configured()
            )
        return SessionResponse(
            authenticated=True,
            auth_configured=is_auth_configured(),
            user=UserSession.model_validate(user),
        )

    @app.post("/api/chat", response_model=chat_service.ChatReply)
    async def chat(payload: ChatRequest, request: Request) -> chat_service.ChatReply:
        require_api_user(request)
        try:
            return chat_service.run_chat(
                payload.question,
                limit=payload.limit,
                agentic=payload.agentic,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Chat request failed: {exc}",
            ) from exc

    return app


def get_session_user(request: Request) -> dict[str, Any] | None:
    user = request.session.get("user")
    if isinstance(user, dict):
        return user
    return None


def require_api_user(request: Request) -> dict[str, Any]:
    user = get_session_user(request)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )
    return user


def is_auth_configured() -> bool:
    return get_entra_settings() is not None


def get_entra_settings() -> EntraSettings | None:
    tenant_id = os.getenv("ENTRA_TENANT_ID")
    client_id = os.getenv("ENTRA_CLIENT_ID")
    client_secret = os.getenv("ENTRA_CLIENT_SECRET")
    if not tenant_id or not client_id or not client_secret:
        return None
    return EntraSettings(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=os.getenv("ENTRA_REDIRECT_URI"),
    )


def get_entra_client():
    settings = get_entra_settings()
    if settings is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Microsoft Entra authentication is not configured.",
        )
    client = oauth.create_client("entra")
    if client is None:
        oauth.register(
            name="entra",
            client_id=settings.client_id,
            client_secret=settings.client_secret,
            server_metadata_url=settings.metadata_url,
            client_kwargs={"scope": "openid profile email"},
        )
        client = oauth.create_client("entra")
    return client


def _redirect_uri(request: Request) -> str:
    settings = get_entra_settings()
    if settings is not None and settings.redirect_uri:
        return settings.redirect_uri
    return str(request.url_for("auth_callback"))


def _claims_to_session_user(claims: dict[str, Any]) -> dict[str, str | None]:
    return {
        "name": claims.get("name")
        or claims.get("preferred_username")
        or "Unknown user",
        "email": claims.get("email") or claims.get("preferred_username"),
        "username": claims.get("preferred_username"),
        "subject": claims.get("sub") or claims.get("oid") or "unknown-subject",
    }


app = create_app()
