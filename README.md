# llms4eu — Web Scraper

A browser-based tool to crawl websites and download their content as `.txt` files.
Built by [Datacation](https://datacation.nl).

## Getting Started

**Requires [uv](https://github.com/astral-sh/uv)**

```bash
# Install dependencies
uv sync

# Start the app (opens on http://localhost:8000)
uv run python llms4eu/main.py
```

## Usage

1. Open `http://localhost:8000` in your browser.
2. Paste one URL per line in the text area.
3. Set the maximum number of pages to crawl per site.
4. Click **Start scraping** — progress updates live.
5. Download the resulting `.txt` files when done.

## Project Structure

```
lllms4eu/
├── llms4eu/
│   ├── __init__.py
│   ├── app.py        # FastAPI web application
│   ├── main.py       # Entry point (starts uvicorn)
│   └── scraper.py    # Crawling and scraping logic
├── data/
│   └── scraped/      # Output .txt files (not committed)
├── pyproject.toml
├── uv.lock
└── README.md
```
