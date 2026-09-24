# Shared

Only what more than one pipeline stage needs.

- `env.py` — repo root, `.env` loading, YAML config reading, and `data_path()`,
  the single resolver for everything under `$LLMS4EU_DATA`.
- `prompts.py` — `render(name, **values)` loads `prompts/<name>.md` and
  substitutes `$placeholders`. The only way prompt text enters the code.
- `llm.py` — structured output from local models via LangChain. Kept
  deliberately: the shared path for any future agentic call, not tied to the
  one caller it has today.

- `wikidata.py` — Wikipedia title to item, item to coordinates, and name search;
  used by page locating (`src/preprocess`) and geo retrieval.

Anything used by a single stage lives in that stage's package instead.
