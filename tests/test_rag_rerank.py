from src.retrieval.retrievers.rerank import _prompt_kwargs


def test_prompt_kwargs_omits_empty_prompt_override():
    assert _prompt_kwargs(None, None) == {}
    assert _prompt_kwargs("rag", None) == {}
    assert _prompt_kwargs(None, "prompt") == {}


def test_prompt_kwargs_builds_custom_prompt_override():
    assert _prompt_kwargs("rag", "prompt") == {
        "prompts": {"rag": "prompt"},
        "default_prompt_name": "rag",
    }
