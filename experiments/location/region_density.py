"""How many candidates does a region leave? Pages of the gold page's category within 10 / 25 km of it.

    uv run python -m experiments.location.region_density

For the gold pages of dev, answer_ok hard questions on `balanced` pages: the number of *other* pages that share a
category with the gold page and lie within the radius (haversine on page coordinates). If a region leaves 5
candidates, resolving it nearly solves the question; if it leaves 500, the specific details still have to do the
work. Metadata only; prints markdown tables.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from src.db.dataset import ROOT

RADII = (10, 25)
EARTH_KM = 6371.0


def haversine_km(
    lat: float, lon: float, lats: np.ndarray, lons: np.ndarray
) -> np.ndarray:
    phi, lam = np.radians(lat), np.radians(lon)
    phis, lams = np.radians(lats), np.radians(lons)
    a = (
        np.sin((phis - phi) / 2) ** 2
        + np.cos(phi) * np.cos(phis) * np.sin((lams - lam) / 2) ** 2
    )
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def main() -> None:
    pages = duckdb.sql(
        f"select id, in_language lang, latitude, longitude, categories from '{ROOT / 'wikipages.parquet'}' where latitude is not null"  # nosec B608 - constant path
    ).df()
    gold = duckdb.sql(
        f"""select distinct id from '{ROOT / "qa" / "wiki_qa_rag.parquet"}'
        where answer_ok and split = 'dev' and kind = 'challenge' and balanced"""  # nosec B608 - constant path
    ).df()
    by_category = {
        c: g
        for c, g in pages.explode("categories").groupby("categories")[
            ["id", "latitude", "longitude"]
        ]
    }
    rows = []
    for g in pages[pages.id.isin(gold.id)].to_dict("records"):
        near = {r: set() for r in RADII}
        for c in g["categories"]:
            same = by_category[c]
            d = haversine_km(
                g["latitude"],
                g["longitude"],
                same.latitude.to_numpy(),
                same.longitude.to_numpy(),
            )
            for r in RADII:
                near[r] |= set(same.id.to_numpy()[d <= r])
        rows.append(
            {"id": g["id"], "lang": g["lang"], "category": g["categories"][0]}
            | {f"within_{r}km": len(near[r] - {g["id"]}) for r in RADII}
        )
    t = pd.DataFrame(rows)

    def summary(group: pd.DataFrame) -> pd.Series:
        out = {"pages": len(group)}
        for r in RADII:
            col = group[f"within_{r}km"]
            out |= {
                f"{r}km_median": col.median(),
                f"{r}km_p90": col.quantile(0.9),
                f"{r}km_le5": (col <= 5).mean(),
            }
        return pd.Series(out)

    print(f"\n{len(t)} gold pages of dev hard questions on balanced pages\n")
    print("### By category (first category of the page)\n")
    print(
        t.groupby("category")
        .apply(summary, include_groups=False)
        .sort_values("pages", ascending=False)
        .round(2)
        .to_markdown()
    )
    print("\n### By language\n")
    print(
        t.groupby("lang")
        .apply(summary, include_groups=False)
        .sort_values("pages", ascending=False)
        .round(2)
        .to_markdown()
    )
    print("\n### All\n")
    print(summary(t).round(2).to_frame("all").T.to_markdown())


if __name__ == "__main__":
    main()
