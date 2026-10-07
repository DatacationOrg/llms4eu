"""Temporary: data tables in the articles (Markdown pipe tables from trafilatura) for computed table questions.
tables(text) -> [{"header": [...], "rows": [[...], ...], "num": {col: [value or None per row]}}]: tables with a header
of >= 2 named columns, >= 4 data rows, and >= 1 column where >= 4 and >= 70% of rows parse as numbers (infoboxes are key/value
tables and fail the header rule). number(cell) parses "1 234,5 m", "1,234.5", "~120[3]" (first number of the cell).
Usage: tables.py survey   (counts per language over all pages)"""

import json
import re
import sys

FOOT = re.compile(r"\[\d+\]|\*+")
NUM = re.compile(r"-?\d[\d\s .,']*")


def number(cell):
    """First number in the cell; decimal comma or point by the usual separator rules; None if none."""
    m = NUM.search(FOOT.sub("", cell))
    if not m:
        return None
    s = re.sub(r"[\s ']", "", m.group()).rstrip(".,")
    if "," in s and "." in s:  # the later one is the decimal mark
        s = (
            s.replace(",", "")
            if s.rfind(".") > s.rfind(",")
            else s.replace(".", "").replace(",", ".")
        )
    elif "," in s:
        s = (
            s.replace(",", ".")
            if re.fullmatch(r"-?\d+,\d{1,2}", s)
            else s.replace(",", "")
        )
    elif s.count(".") > 1 or re.fullmatch(r"-?\d{1,3}\.\d{3}", s):
        s = s.replace(
            ".", ""
        )  # 1.234 / 1.234.567 = thousands (ambiguous 1.234 read as 1234)
    try:
        return float(s)
    except ValueError:
        return None


def cells(line):
    return [FOOT.sub("", c).strip() for c in line.strip().strip("|").split("|")]


def tables(text):
    out, block = [], []
    for line in text.split("\n") + [""]:
        if line.lstrip().startswith("|"):
            block.append(line)
            continue
        if len(block) >= 6 and re.fullmatch(r"\|?[\s:|-]+\|?", block[1].strip()):
            header = cells(block[0])
            rows = [cells(x) for x in block[2:]]
            rows = [r for r in rows if len(r) == len(header) and any(r)]
            num = {}
            for c in range(1, len(header)):
                vals = [number(r[c]) for r in rows]
                if sum(v is not None for v in vals) >= max(
                    4, 0.7 * len(rows)
                ):  # infobox value columns mix words
                    num[c] = vals
            if sum(bool(h) for h in header) >= 2 and len(rows) >= 4 and num:
                out.append({"header": header, "rows": rows, "num": num})
        block = []
    return out


if __name__ == "__main__":
    assert (
        number("1 234,5 m") == 1234.5
        and number("1,234.5") == 1234.5
        and number("~120[3]") == 120
    )
    assert number("12,5") == 12.5 and number("1.234") == 1234 and number("n/a") is None
    if sys.argv[1:] == ["survey"]:
        import collections

        from bunny import PAGES

        c = collections.Counter()
        for line in open(PAGES):
            p = json.loads(line)
            t = tables(p["text"])
            c[p["in_language"], "pages"] += bool(t)
            c[p["in_language"], "tables"] += len(t)
        langs = sorted({k[0] for k in c}, key=lambda x: -c[x, "pages"])
        print({lang: (c[lang, "pages"], c[lang, "tables"]) for lang in langs})
        print(
            "total pages",
            sum(c[x, "pages"] for x in langs),
            "tables",
            sum(c[x, "tables"] for x in langs),
        )
