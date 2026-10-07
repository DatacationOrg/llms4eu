"""Click through a sample of the wiki QA set and judge it by hand.

    uv run --group notebook python notebooks/review_app.py [port]   # open the URL it prints

The sample is drawn once into out/review_items.json (delete it for a new one); every click is appended to
out/review_labels.jsonl, the last answer per item and check wins. explore_wiki_qa.ipynb reads the labels back.
"""

from __future__ import annotations

import json
import sys
import zlib
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import duckdb

QA = Path("/data/llms4eu/wiki/qa")
PAGES = QA.parent / "pages.jsonl"
OUT = Path(__file__).parent / "out"
ITEMS, LABELS = OUT / "review_items.json", OUT / "review_labels.jsonl"
SEED = 7

# (stratum, language, file, filter, n): Dutch first, English second, translations judged in the target language
PLAN = [
    ("easy", "nl", "rag", "answer_ok and kind = 'corpus'", 10),
    ("hard", "nl", "rag", "answer_ok and kind = 'challenge'", 10),
    ("ambiguous", "nl", "rag", "answer_ok and hits in ('few', 'many')", 4),
    ("rejected", "nl", "rag", "answer_ok = false", 6),
    ("translation", "nl", "rag", "answer_ok and x_ok and lang not in ('nl', 'en')", 8),
    ("unanswerable", "nl", "unanswerable", "ok", 6),
    ("compare", "nl", "compare", "ok", 4),
    ("meta", "nl", "meta", "ok", 3),
    ("easy", "en", "rag", "answer_ok and kind = 'corpus'", 4),
    ("hard", "en", "rag", "answer_ok and kind = 'challenge'", 6),
    ("translation", "en", "rag", "answer_ok and x_ok and lang not in ('nl', 'en')", 4),
]
# Blind check of the dataset's own `realistic` label on hard questions: half true, half false, label not shown.
# Added after the first sample, so these go in front of it (see main); the notebook compares verdicts with the label.
REALISM = "answer_ok and kind = 'challenge' and (labels->>'realistic') = "
PLAN += [
    ("realism", lang, "rag", REALISM + f"'{flag}'", n)
    for lang, n in [("nl", 6), ("en", 4)]
    for flag in ["true", "false"]
]

# what to judge per stratum: (key, question shown, how to decide). Judge against the page, not the real world.
NATURAL = (
    "natural",
    "Would a real person ask this?",
    "Anyone at all counts here: a tourist, a local, a historian, a quiz fan. Yes: some real person could type or say "
    "this to a search engine or chatbot, in this form. Hard questions are meant to be vague, as by someone who "
    "half-remembers the place. No: it reads like a riddle built from the page rather than a question anyone has: "
    "mentions 'the article', asks for codes or register data, stacks details only a reader of the page would know, "
    "or gives the answer away. Example: a question about a 1620 land grant is natural (a historian asks it), "
    "but not a tourist question.",
)
FLUENT = (
    "fluent",
    "Fluent and correct language?",
    "Yes: a native speaker could have written it; Belgian Dutch wording is fine. No: grammar or spelling errors, "
    "garbled or invented words, words from another language, or a stiff literal translation. Ignore style taste.",
)
UNIQUE = (
    "unique",
    "Points to this one place?",
    "Yes: the name plus locator (easy) or the combination of details (hard) fits this place and no other EU place "
    "of this type you can think of. No: another place plausibly fits as well, e.g. a common lake name in the same "
    "municipality, or 'a 12th-century castle in Germany'. Do not search exhaustively; unsure if you cannot tell.",
)
ANSWER = (
    "answer",
    "Answer correct and in the page?",
    "Yes: it answers what was asked, every part of it, and the page supports it. Check the highlighted evidence; "
    "open the full text when the evidence does not cover a detail. Extra details are fine if they are in the "
    "page. No: a wrong or unsupported fact, answers a different question, or misses part of a two-part question. "
    "If the page contradicts what you know, judge by the page and write it in the notes.",
)
CHECKS = {
    "easy": [NATURAL, FLUENT, UNIQUE, ANSWER],
    "hard": [NATURAL, FLUENT, UNIQUE, ANSWER],
    "ambiguous": [
        NATURAL,
        FLUENT,
        (
            "alternatives",
            "Do the listed other pages really fit too?",
            "Open each other page. Yes: they fit the question as well as the gold page, so it is truly ambiguous. "
            "No: at least one clearly does not fit (the LLM judge was wrong). Some fit, some not: unsure, say "
            "which in the notes.",
        ),
    ],
    "rejected": [
        NATURAL,
        FLUENT,
        ANSWER,
        (
            "judge_right",
            "Was rejecting it right?",
            "See 'Failed checks'. Yes: the answer really has that problem. No: the answer is fine and the judge "
            "was too strict, e.g. the detail is in the page but outside the quoted evidence.",
        ),
    ],
    "translation": [
        (
            "faithful",
            "Translation means the same?",
            "Compare with the original via the English line. Yes: same question, place names kept correctly, "
            "no constraint added or dropped. No: the meaning changed, a name is mistranslated, or a detail "
            "is lost.",
        ),
        (
            "natural",
            "Translation sounds natural?",
            "Yes: a native speaker of the translation language would ask it this way. No: understandable but "
            "clearly translated, odd word order, wrong words.",
        ),
    ],
    "unanswerable": [
        NATURAL,
        FLUENT,
        (
            "unanswerable",
            "Really not answerable from the page?",
            "Read 'Why', then search the full page text. Yes: the page does not answer it (not_covered), or "
            "it contradicts the premise (false_premise). No: the page does answer it, or the premise is true. "
            "Note it if the question is a stock template, e.g. ticket prices or opening hours.",
        ),
    ],
    "compare": [
        NATURAL,
        FLUENT,
        (
            "answer",
            "Answer correct for both places?",
            "Yes: both facts match their pages and the comparison or arithmetic is right. No: a fact is wrong for "
            "either place, or the conclusion is wrong.",
        ),
    ],
    "realism": [
        NATURAL,
        FLUENT,
        (
            "tourist",
            "Would a tourist ask this?",
            "Narrower than the previous check: only our user counts, a tourist choosing where to go, planning a visit, "
            'standing on site, or curious after the trip. Yes: it fits one of those moments ("what is this castle?", '
            '"how high is it?", "which castle did we see near X?"). No: only another persona would ask it (a quiz '
            "writer, a historian, a hydrologist), even if the question itself is natural. Optional.",
        ),
        (
            "clues",
            "Could someone know these clues without reading the page?",
            "Yes: the clues are things a visitor could see, hear or remember: region, type of place, what it looks like, "
            "a well-known story, a rough age. No: they are things you only know from the page: exact numbers or dates, "
            "register data, administrative classifications, exact distances. Note the clue that decided it.",
        ),
    ],
    "meta": [
        NATURAL,
        FLUENT,
        (
            "gold",
            "Gold list looks right and complete?",
            "Yes: every listed place is of the asked type and inside the area. No: a listed place is clearly wrong, or "
            "a place you know (or see in the anchor page) is clearly missing. Completeness cannot be fully checked: "
            "unsure is fine.",
        ),
    ],
}

GUIDE = """<ul>
<li><b>Judge against the page</b>, not the real world: the test is whether a system using this page would be right.</li>
<li><b>Unsure is a real answer.</b> Use it instead of guessing; it is counted apart.</li>
<li><b>Every no gets a short note</b> starting with a tag: <i>contrived, overspecified (more detail than a person would give; fine to keep yes), leak, stiff (correct but unidiomatic), grammar, garbled, wrong fact,
unsupported, incomplete, ambiguous: [other place], bad translation, template</i>. Tags make the notes countable.</li>
<li><b>About a minute per item.</b> Decide and move on; do not go back to change earlier items.</li>
<li>Judge each check on its own: a fluent question can still be unnatural, and a natural one can have a wrong answer.</li>
</ul>"""


def build(plan=PLAN, exclude: frozenset = frozenset()) -> list[dict]:
    """The items of `plan`, skipping the item keys in `exclude` (already in the sample)."""
    con = duckdb.connect()
    for f in ["rag", "unanswerable", "compare", "meta", "clean", "challenge"]:
        con.sql(f"create view {f} as select * from '{QA}/wiki_qa_{f}.parquet'")
    con.sql(
        f"create table pages as select * exclude (text) from read_json_auto('{PAGES}')"
    )
    con.sql("""create table gloss as select id, q, any_value(q_en) q_en from (
        select id, item->>'question' q, item->>'question_en' q_en from clean
        union all select id, repaired->>'question', repaired->>'question_en' from clean where starts_with(repaired, '{')
        union all select id, item->>'question', item->>'question_en' from challenge) group by all""")
    items = []
    for stratum, lang, file, where, n in plan:
        by_lang = (
            f"x_lang = '{lang}'" if stratum == "translation" else f"lang = '{lang}'"
        )
        if file == "rag":
            sql = f"""select r.*, g.q_en question_en from rag r left join gloss g on g.id = r.id and g.q = r.question
                      where {by_lang} and {where} and not list_contains($skip, 'rag|' || r.id || '|' || r.n)
                      order by hash(r.id || r.n || {SEED}) limit {n}"""
        else:
            key = "key" if file == "meta" else "id"
            sql = f"select * from {file} where {by_lang} and {where} order by hash({key} || question || {SEED}) limit {n}"
        params = {"skip": sorted(exclude)} if file == "rag" else None
        for row in con.execute(sql, params).df().to_dict("records"):
            items.append(item(con, stratum, lang, file, row))
    texts = dict(
        con.execute(
            f"select id, text from read_json_auto('{PAGES}') where list_contains(?, id)",
            [sorted({p["id"] for i in items for p in i["pages"]})],
        ).fetchall()
    )
    for i in items:
        for p in i["pages"]:
            p["text"] = texts.get(p["id"], "")[:20000]
    return items


def refs(con, ids, spans=()) -> list[dict]:
    """The pages to show for an item: title, link and the evidence spans to highlight (texts are added later)."""
    meta = {
        i: (t, u)
        for i, t, u in con.execute(
            "select id, title, url from pages where list_contains(?, id)", [list(ids)]
        ).fetchall()
    }
    spans = list(spans) or [[]] * len(ids)
    return [
        {
            "id": i,
            "title": meta.get(i, (i, ""))[0],
            "url": meta.get(i, ("", ""))[1],
            "spans": [list(map(int, x)) for x in sp],
        }
        for i, sp in zip(ids, spans)
    ]


def item(con, stratum, lang, file, r) -> dict:
    """One row as the review page shows it: its texts, its page(s) with evidence, and the checks to answer."""
    it = {"stratum": stratum, "lang": lang, "file": file}
    if file == "rag":
        it["key"] = f"rag|{r['id']}|{r['n']}"
        if stratum == "translation":
            it["fields"] = [
                (f"Translation ({r['x_lang']})", r["question_x"]),
                (f"Original ({r['lang']})", r["question"]),
            ]
        else:
            it["fields"] = [("Question", r["question"])]
        it["fields"] += [("English", r["question_en"]), ("Answer", r["answer"])]
        if stratum == "rejected":
            crit = json.loads(r["criteria"]) if r["criteria"] else {}
            it["fields"].append(
                (
                    "Failed checks",
                    ", ".join(k for k, v in crit.items() if v is False) or "not judged",
                )
            )
        others = (
            list(r["relevant"])
            if stratum == "ambiguous" and r["relevant"] is not None
            else []
        )
        it["pages"] = refs(con, [r["id"]], [r["spans"]]) + refs(con, others[:5])
    elif file == "unanswerable":
        it["key"] = f"unans|{r['id']}|{r['type']}"
        it["fields"] = [
            ("Question", r["question"]),
            ("English", r["question_en"]),
            ("Type", r["type"]),
            ("Why", r["why"]),
        ]
        it["pages"] = refs(con, [r["id"]])
    elif file == "compare":
        it["key"] = f"compare|{'|'.join(r['pages'])}"
        it["fields"] = [
            ("Question", r["question"]),
            ("English", r["question_en"]),
            ("Answer", r["answer"]),
        ]
        it["pages"] = refs(con, list(r["pages"]), [r["spans_a"], r["spans_b"]])
    else:
        it["key"] = f"meta|{r['key']}"
        it["fields"] = [
            ("Question", r["question"]),
            ("English", r["question_en"]),
            ("Task", r["spec"]),
        ]
        it["pages"] = refs(
            con, ([r["anchor"]] if r["anchor"] else []) + list(r["gold_pages"])
        )
        it["fields"].append(
            (
                "Gold places",
                ", ".join(p["title"] for p in it["pages"][1 if r["anchor"] else 0 :]),
            )
        )
    return it


def labels() -> dict:
    out = {}
    if LABELS.exists():
        for line in LABELS.read_text().splitlines():
            rec = json.loads(line)
            out.setdefault(rec["key"], {})[rec["check"]] = rec["value"]
    return out


class Handler(BaseHTTPRequestHandler):
    def send(self, body: str, kind="application/json"):
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", f"{kind}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/items":
            self.send(ITEMS.read_text())
        elif (
            self.path == "/api/checks"
        ):  # served live, so editing a rule needs no new sample
            self.send(json.dumps(CHECKS))
        elif self.path == "/api/labels":
            self.send(json.dumps(labels()))
        else:
            self.send(PAGE.replace("<!--GUIDE-->", GUIDE), "text/html")

    def do_POST(self):
        rec = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        rec["at"] = datetime.now().isoformat(timespec="seconds")
        with LABELS.open("a") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.send("{}")

    def log_message(self, *args):
        pass


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Wiki QA review</title><style>
body{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:0 auto;padding:16px;color:#222;background:#fafafa}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.tag{background:#e8eef7;border-radius:4px;padding:2px 8px;font-size:13px}
.f{margin:10px 0}.f b{display:block;font-size:12px;color:#666;text-transform:uppercase}
.check{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:6px 0;border-bottom:1px solid #eee}
.rule{display:block;font-size:12px;color:#666}.guide{background:#fff;border:1px solid #ddd;padding:6px 10px;margin:10px 0}
.check>span:last-child{white-space:nowrap}
.check button{margin-left:4px;padding:4px 12px;border:1px solid #bbb;background:#fff;border-radius:4px;cursor:pointer}
.check button.on{background:#2b6cb0;color:#fff;border-color:#2b6cb0}
mark{background:#fde68a}.ex{font-size:13px;background:#fff;border:1px solid #ddd;padding:8px;margin:6px 0;white-space:pre-wrap}
textarea{width:100%;height:60px}nav{display:flex;justify-content:space-between;margin-top:16px}
nav button{padding:6px 16px}details pre{white-space:pre-wrap;font-size:12px;max-height:400px;overflow:auto;background:#fff}
</style></head><body>
<header><select id="group" onchange="pick(this.value)"></select><b id="pos"></b><span class="tag" id="stratum"></span><span class="tag" id="lang"></span><span id="done"></span></header>
<details class="guide"><summary>How to judge</summary><!--GUIDE--></details>
<div id="fields"></div><h4>Checks</h4><div id="checksBox"></div>
<div class="f"><b>Notes</b><textarea id="notes" placeholder="anything odd: wrong fact, contrived, not real Dutch..."></textarea></div>
<div id="pages"></div>
<nav><button onclick="go(-1)">&larr; prev (k)</button><button onclick="go(1)">next (j) &rarr;</button></nav>
<script>
let all=[], items=[], checks={}, labels={}, i=0;
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function load(){all=await (await fetch('/api/items')).json();checks=await (await fetch('/api/checks')).json();labels=await (await fetch('/api/labels')).json();
  pick(location.hash.slice(1)||'')}
// one group at a time (#realism in the URL), or everything; opens at the group's first unjudged item
function pick(g){location.hash=g;items=g?all.filter(t=>t.stratum===g):all;
  i=Math.max(0,items.findIndex(t=>!labels[t.key]));render()}
function groups(){const n={};for(const t of all){n[t.stratum]=n[t.stratum]||[0,0];n[t.stratum][1]++;if(labels[t.key])n[t.stratum][0]++}
  const g=location.hash.slice(1);
  group.innerHTML=`<option value="">all groups (${all.filter(t=>labels[t.key]).length}/${all.length})</option>`+
    Object.entries(n).map(([k,[d,c]])=>`<option value="${k}" ${k===g?'selected':''}>${k} (${d}/${c})</option>`).join('')}
function save(check,value){const k=items[i].key;(labels[k]=labels[k]||{})[check]=value;
  fetch('/api/label',{method:'POST',body:JSON.stringify({key:k,stratum:items[i].stratum,lang:items[i].lang,check,value})});render(true)}
function excerpt(text,[s,e]){return esc(text.slice(Math.max(0,s-300),s))+'<mark>'+esc(text.slice(s,e))+'</mark>'+esc(text.slice(e,e+300))}
function render(keepNotes){const t=items[i],l=labels[t.key]||{};
  pos.textContent=`${i+1} / ${items.length}`;stratum.textContent=t.stratum;lang.textContent=t.lang;
  done.textContent='';groups();
  fields.innerHTML=t.fields.map(([k,v])=>`<div class="f"><b>${esc(k)}</b>${esc(v)}</div>`).join('');
  checksBox.innerHTML=checks[t.stratum].map(([k,q,rule])=>`<div class="check"><span>${esc(q)}<span class="rule">${esc(rule)}</span></span><span>`+
    ['yes','no','unsure'].map(v=>`<button class="${l[k]===v?'on':''}" onclick="save('${k}','${v}')">${v}</button>`).join('')+'</span></div>').join('');
  if(!keepNotes){notes.value=l.notes||''}
  pages.innerHTML=t.pages.map(p=>`<h4>${esc(p.title)} <small>${esc(p.id)}</small> ${p.url?`<a href="${esc(p.url)}" target="_blank">open</a>`:''}</h4>`+
    p.spans.map(sp=>`<div class="ex">…${excerpt(p.text,sp)}…</div>`).join('')+
    `<details><summary>full page text</summary><pre>${esc(p.text)}</pre></details>`).join('')}
function go(d){if(notes.value!==((labels[items[i].key]||{}).notes||''))save('notes',notes.value);
  i=Math.min(items.length-1,Math.max(0,i+d));render();scrollTo(0,0)}
document.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA')return;if(e.key==='j')go(1);if(e.key==='k')go(-1)});
load();
</script></body></html>"""

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if not ITEMS.exists():
        print("drawing the sample ...")
        ITEMS.write_text(json.dumps(build(), ensure_ascii=False, default=str))
    # strata added to PLAN later are drawn once and put first, so the app opens on them; judged items keep their labels
    items = json.loads(ITEMS.read_text())
    have = {(i["stratum"], i["lang"]) for i in items}
    if new := [p for p in PLAN if (p[0], p[1]) not in have]:
        print(f"drawing {len(new)} new strata ...")
        keys = {i["key"] for i in items}
        added = build(new, frozenset(keys))
        added.sort(
            key=lambda i: zlib.crc32(i["key"].encode())
        )  # mix true / false so the order gives nothing away
        ITEMS.write_text(json.dumps(added + items, ensure_ascii=False, default=str))
    # a free port by default: the server is shared, fixed ports collide
    server = ThreadingHTTPServer(
        ("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 0), Handler
    )
    print(
        f"{len(json.loads(ITEMS.read_text()))} items; open http://localhost:{server.server_port}",
        flush=True,
    )
    server.serve_forever()
