# RAG

Retrieves places and answers questions.

Search embeds the query, asks Chroma for the top configured matches, then fetches
full rows from SQLite by `id`.

`answer.py` prints retrieved rows, prompt context, and the local Gemma response.
Retrieval config lives in `config.yaml`; answer config lives in
`answer_config.yaml`.

`limit=3` means: return the top 3 Chroma vector matches by cosine similarity.
Chroma returns scores and ids; SQLite returns the full text rows.

Ollama keeps model serving local. LangChain structured output plus Pydantic keeps
the response shaped as data.

Retrieve from Chroma, then fetch rows from SQLite:

```python
from src.rag.search import search_places

places = search_places("quiet forest walk near water", limit=3)
print(places[0].score, places[0].summary)
```

Current search flow:

```text
query text -> MiniLM embedding -> Chroma top-k search -> ids -> SQLite rows
```

Vector calls live in `src/vector_db/places.py`.

Make a structured local LangChain call:

```python
from pydantic import BaseModel

from src.shared.llm import structured_local_model


class Answer(BaseModel):
    answer: str


model = structured_local_model("gemma4:e4b", Answer)
result = model.invoke("Answer briefly: what is a calm lake good for?")
print(result.answer)
```
