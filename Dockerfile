FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ADD ./pyproject.toml ./uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

ADD llms4eu ./llms4eu/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev
ENTRYPOINT ["uv", "run", "python", "/app/llms4eu/main.py"]
