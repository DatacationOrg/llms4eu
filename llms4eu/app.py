import threading
import uuid
from pathlib import Path
from typing import TypedDict

from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel

from llms4eu.scraper import SiteResult, save_results_as_txt, scrape_urls

app = FastAPI(title="Web Scraper")

OUTPUT_DIR = Path("data/scraped")


class SiteInfo(TypedDict):
    status: str
    message: str


class Job(TypedDict):
    status: str
    sites: dict[str, SiteInfo]
    files: list[str]
    error: str | None


class ScrapeRequest(BaseModel):
    urls: list[str]
    max_pages: int = 50


# In-memory job storage: job_id -> job dict
jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Web Scraper — Datacation</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    body { font-family: 'Inter', sans-serif; }
    .spinner {
      border: 3px solid #e5e7eb;
      border-top-color: #7c3aed;
      border-radius: 50%;
      width: 20px;
      height: 20px;
      animation: spin 0.8s linear infinite;
      display: inline-block;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body class="bg-gray-50 min-h-screen">

  <!-- Header -->
  <header class="bg-white border-b border-gray-200 px-6 py-4">
    <div class="max-w-3xl mx-auto flex items-center gap-4">
      <img
        src="https://www.datacation.nl/assets/standard.svg"
        alt="Datacation"
        class="h-8"
        onerror="this.style.display='none'; document.getElementById('logo-fallback').style.display='block';"
      />
      <span id="logo-fallback" style="display:none"
        class="text-xl font-bold text-purple-700 tracking-tight">Datacation</span>
      <span class="text-gray-300 text-xl font-light">|</span>
      <span class="text-gray-600 font-medium">Web Scraper</span>
    </div>
  </header>

  <!-- Main -->
  <main class="max-w-3xl mx-auto px-6 py-10">

    <div class="bg-white rounded-2xl shadow-sm border border-gray-200 p-8">

      <h1 class="text-2xl font-semibold text-gray-900 mb-1">Scrape websites to text</h1>
      <p class="text-gray-500 mb-8 text-sm">
        Paste one URL per line. Each site will be crawled and saved as a <code>.txt</code> file.
      </p>

      <!-- URL input -->
      <div class="mb-5">
        <label class="block text-sm font-medium text-gray-700 mb-2">Website URLs</label>
        <textarea
          id="urls"
          rows="6"
          class="w-full border border-gray-300 rounded-xl p-3 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-y"
          placeholder="https://example.com&#10;https://another-site.com"
        ></textarea>
      </div>

      <!-- Max pages -->
      <div class="mb-8 flex items-center gap-4">
        <label class="text-sm font-medium text-gray-700 whitespace-nowrap">Max pages per site</label>
        <input
          type="number"
          id="max-pages"
          value="50"
          min="1"
          max="500"
          class="border border-gray-300 rounded-lg px-3 py-2 w-24 text-sm focus:outline-none focus:ring-2 focus:ring-purple-500"
        />
        <span class="text-xs text-gray-400">Higher = more thorough but slower</span>
      </div>

      <!-- Scrape button -->
      <button
        id="scrape-btn"
        onclick="startScrape()"
        class="bg-purple-600 hover:bg-purple-700 active:bg-purple-800 text-white px-6 py-3 rounded-xl font-semibold text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        Start scraping
      </button>
    </div>

    <!-- Progress & results panel -->
    <div id="panel" class="hidden mt-6 bg-white rounded-2xl shadow-sm border border-gray-200 p-8">

      <!-- Status row -->
      <div id="status-row" class="flex items-center gap-3 mb-6">
        <div id="spinner" class="spinner hidden"></div>
        <span id="status-text" class="text-sm font-medium text-gray-700"></span>
      </div>

      <!-- Per-site progress -->
      <div id="sites-list" class="space-y-3 mb-6"></div>

      <!-- Downloads -->
      <div id="downloads" class="hidden">
        <h2 class="text-sm font-semibold text-gray-700 mb-3 uppercase tracking-wide">
          Download results
        </h2>
        <div id="download-links" class="space-y-2"></div>
      </div>
    </div>

  </main>

  <script>
    let jobId = null;
    let pollTimer = null;

    async function startScrape() {
      const rawUrls = document.getElementById('urls').value;
      const maxPages = parseInt(document.getElementById('max-pages').value, 10) || 50;
      const urls = rawUrls.split('\\n').map(u => u.trim()).filter(Boolean);

      if (urls.length === 0) {
        alert('Please enter at least one URL.');
        return;
      }

      // Reset UI
      clearInterval(pollTimer);
      document.getElementById('panel').classList.remove('hidden');
      document.getElementById('downloads').classList.add('hidden');
      document.getElementById('download-links').innerHTML = '';
      document.getElementById('sites-list').innerHTML = '';
      document.getElementById('scrape-btn').disabled = true;
      setStatus('Starting…', true);

      const resp = await fetch('/scrape', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ urls, max_pages: maxPages }),
      });
      const data = await resp.json();
      jobId = data.job_id;
      pollTimer = setInterval(poll, 1500);
    }

    async function poll() {
      if (!jobId) return;
      const resp = await fetch(`/status/${jobId}`);
      const job = await resp.json();

      // Update per-site progress
      const sitesList = document.getElementById('sites-list');
      sitesList.innerHTML = '';
      for (const [siteUrl, info] of Object.entries(job.sites || {})) {
        const done = info.status === 'done';
        const error = info.status === 'error';
        const color = done ? 'text-green-600' : error ? 'text-red-500' : 'text-purple-600';
        const icon = done ? '✓' : error ? '✗' : '…';
        sitesList.innerHTML += `
          <div class="flex items-start gap-3 text-sm">
            <span class="font-mono font-bold ${color} mt-0.5">${icon}</span>
            <div>
              <div class="font-medium text-gray-800">${siteUrl}</div>
              <div class="text-gray-500 text-xs">${info.message || ''}</div>
            </div>
          </div>`;
      }

      if (job.status === 'done') {
        clearInterval(pollTimer);
        setStatus('Scraping complete!', false);
        showDownloads(job.files);
        document.getElementById('scrape-btn').disabled = false;
      } else if (job.status === 'error') {
        clearInterval(pollTimer);
        setStatus(`Error: ${job.error}`, false);
        document.getElementById('scrape-btn').disabled = false;
      } else {
        setStatus('Scraping in progress…', true);
      }
    }

    function setStatus(msg, spin) {
      document.getElementById('status-text').textContent = msg;
      document.getElementById('spinner').classList.toggle('hidden', !spin);
    }

    function showDownloads(files) {
      const container = document.getElementById('download-links');
      container.innerHTML = '';
      for (const f of files) {
        const a = document.createElement('a');
        a.href = `/download/${jobId}/${f}`;
        a.download = f;
        a.className =
          'flex items-center gap-2 px-4 py-2 bg-purple-50 hover:bg-purple-100 ' +
          'border border-purple-200 rounded-lg text-sm font-medium text-purple-700 transition-colors w-fit';
        a.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" fill="none"
          viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
          <path stroke-linecap="round" stroke-linejoin="round"
            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
          ${f}`;
        container.appendChild(a);
      }
      document.getElementById('downloads').classList.remove('hidden');
    }
  </script>

</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return HTML


@app.post("/scrape")
async def start_scrape(
    data: ScrapeRequest, background_tasks: BackgroundTasks
) -> dict[str, str]:
    urls: list[str] = data.urls
    max_pages: int = data.max_pages

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        jobs[job_id] = {
            "status": "pending",
            "sites": {url: {"status": "pending", "message": "Queued"} for url in urls},
            "files": [],
            "error": None,
        }

    background_tasks.add_task(_run_scrape, job_id, urls, max_pages)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str) -> JSONResponse:
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse(job)


@app.get("/download/{job_id}/{filename}", response_model=None)
async def download_file(job_id: str, filename: str) -> FileResponse | JSONResponse:
    # Prevent path traversal
    safe_name = Path(filename).name
    filepath = OUTPUT_DIR / job_id / safe_name
    if not filepath.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(str(filepath), filename=safe_name, media_type="text/plain")


def _run_scrape(job_id: str, urls: list[str], max_pages: int) -> None:
    with _jobs_lock:
        jobs[job_id]["status"] = "running"

    def on_progress(site_url: str, page_url: str, visited: int, total: int) -> None:
        with _jobs_lock:
            jobs[job_id]["sites"][site_url] = {
                "status": "running",
                "message": f"Page {visited}/{total}: {page_url}",
            }

    try:
        results: list[SiteResult] = scrape_urls(
            urls, max_pages=max_pages, on_progress=on_progress
        )

        # Mark site statuses
        for result in results:
            with _jobs_lock:
                if result.error:
                    jobs[job_id]["sites"][result.site_url] = {
                        "status": "error",
                        "message": result.error,
                    }
                else:
                    jobs[job_id]["sites"][result.site_url] = {
                        "status": "done",
                        "message": f"{len(result.pages)} pages scraped",
                    }

        output_dir = OUTPUT_DIR / job_id
        saved = save_results_as_txt(results, output_dir=str(output_dir))

        with _jobs_lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["files"] = [f.name for f in saved]

    except (OSError, ValueError, RuntimeError) as exc:
        logger.exception("Scrape job failed")
        with _jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(exc)
