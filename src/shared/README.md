# Shared

Tiny helpers used by more than one script.

`schema.py` is the Python data contract, `embed.py` wraps SentenceTransformer
embedding calls, `indexers.py` exposes buildable provider-shaped embedding
backends,
`env.py` loads local settings, `cli.py` keeps shared command-line prompts small,
and `llm.py` exposes structured-output clients.

Keep this folder conservative: add code here only after more than one service
needs it.

Typical imports:

```python
from src.shared.schema import Place
from src.shared.embed import embed_texts
from src.shared.indexers import build_indexer
from src.shared.cli import confirm
```
