from dataclasses import dataclass


@dataclass(frozen=True)
class RenderedPage:
    final_url: str
    title: str
    html: str


def render_html(url: str, timeout_ms: int = 45_000) -> RenderedPage:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)
            rendered = RenderedPage(
                final_url=page.url,
                title=page.title(),
                html=page.content(),
            )
        finally:
            browser.close()
    return rendered
