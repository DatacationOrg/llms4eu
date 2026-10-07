# Question-data cleaning rules

The data is training data for a small model that writes RAG test questions. Each article comes
from a corpus of ~100k Wikipedia pages about European places (lakes, castles, mountains, caves,
reserves…) in 24 languages. A chat user who has never seen the article asks a question and the
search system has to find this page among all the pages. Treat every item as a gold example:
the small model copies whatever you leave in, mistakes included.

Input `batch.jsonl`: one article per line, `{url, lang, title, country, text, items}`.
Output `fixed.jsonl`: one line per input article, same order, `{"url": ..., "items": [...]}`.
Each item has exactly these keys, in this order:
`{"question": str, "question_en": str, "facts": [str, 1-3], "query": str}`.

## Rules

1. **Only questions a real person would ask.** Things a curious visitor, hiker, student or
   traveller would type into a chat assistant: what the place is like, what there is to see or
   do, its history, owners, legends, nature and wildlife, size, height, depth, how to get there,
   trails, opening/visiting. Delete any question about codes, IDs, register numbers, basin or
   catchment numbers, coordinates, map sheets, or other database fields, and trivia nobody
   would ask ("Which drainage basin number…", "What are the WGS 84 coordinates…").
2. **Scale to the article.** A short stub gets 1-2 questions, a rich article up to 5. No two
   questions may ask the same thing or share the same answer. If there is nothing worth asking,
   return `"items": []`. Fewer strong questions beat five weak ones.
3. **Unique among 100k pages.** Many places share names (hundreds of lakes named the same).
   Each question must name the place together with a distinguishing detail (municipality,
   region, mountain range, nearby town) or describe it so that only this place fits. The
   question may describe the place instead of naming it, as long as only this place fits.
4. **Language and fluency.** `question` is in the article's language (`lang`), fluent and
   grammatical, with correct inflection of names (e.g. Finnish "Haukilammen", Latvian
   "lielākais dziļums"). Never refer to "the article"/"the text"; never put the answer in the
   question. Prefer paraphrase over copying the article's wording.
5. **question_en**: a faithful, natural English translation of `question`.
6. **facts**: 1-3 short facts in the article's language, each stated in or directly supported
   by the article text, that answer the question (not restate it). Check every fact against the
   text; fix garbled words; drop the item if the answer is not in the text.
7. **query**: English search keywords an embedder would use to find this page, built only from
   information in the question. It must not contain the answer (no numbers, names or words from
   the facts that are not in the question). Place names stay as written in the question.

Fix what can be fixed (grammar, disambiguation, translation, a leaking query); rewrite a weak
question into a strong one if the article supports it; otherwise delete it. You may add a new
question if the article has an obviously interesting fact that no item covers and the article
has room under rule 2.

## Process

Work through the batch yourself: read each article and its items, decide, write. Do not call
external APIs or models. You may write small helper scripts, but only inside your own folder.
When done, run `python3 /home/gergely/work/llms4eu/tmp_labeling/fix2_check.py <your folder>`
and fix everything it reports until it prints OK. Report: articles, items in → items out, and
the main kinds of changes (a few lines).
