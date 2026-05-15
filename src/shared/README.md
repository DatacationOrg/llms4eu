# Shared

Tiny helpers used by more than one script.

`schema.py` is the Python data contract, `embed.py` wraps the local embedding
model, `env.py` loads local settings, and `llm.py` creates the local
structured-output model.

Keep this folder conservative: add code here only after more than one service
needs it.

Typical imports:

```python
from src.shared.schema import Place
from src.shared.embed import embed_texts
```
