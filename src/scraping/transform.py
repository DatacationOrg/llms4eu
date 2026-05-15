import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from src.scraping.scraper import PageContent, SiteResult
from src.shared.env import load_yaml
from src.shared.llm import structured_local_model
from src.shared.schema import Place


CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))


class PlaceSummary(BaseModel):
    summary: str = Field(description="A short factual tourism summary.")


def transform_results(
    results: list[SiteResult],
    place_per: str | None = None,
    summary_model: str | None = None,
    summarizer=None,
) -> list[Place]:
    mode = place_per or CONFIG["place_per"]
    summary_model = summary_model or CONFIG["summary_model"]
    summarizer = summarizer or structured_local_model(summary_model, PlaceSummary)

    places: list[Place] = []
    for site in results:
        if site.error:
            continue
        if mode == "site":
            place = _site_place(site, summarizer)
            if place:
                places.append(place)
            continue
        for page in site.pages:
            place = _page_place(page, summarizer)
            if place:
                places.append(place)
    return places


def _page_place(page: PageContent, summarizer) -> Place | None:
    description = _page_description(page)
    if not description:
        return None
    return Place(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, page.url)),
        place_description=description,
        summary=_summarize(description, page.url, summarizer),
    )


def _site_place(site: SiteResult, summarizer) -> Place | None:
    description = "\n\n".join(
        part for part in (_page_description(page) for page in site.pages) if part
    )
    if not description:
        return None
    return Place(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, site.site_url)),
        place_description=description,
        summary=_summarize(description, site.site_url, summarizer),
    )


def _page_description(page: PageContent) -> str:
    parts = [page.title.strip(), page.meta_description.strip(), *page.paragraphs]
    return "\n\n".join(part for part in parts if part)


def _summarize(description: str, url: str, summarizer) -> str:
    prompt = (
        "Write one short factual summary for a tourism place based only on the text below. "
        "Do not invent details. Keep it under 35 words.\n\n"
        f"url: {url}\n\ntext:\n{description}"
    )
    return summarizer.invoke(prompt).summary.strip()
