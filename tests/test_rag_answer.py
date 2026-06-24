from src.rag.answer import RagAnswer, answer_question


def test_answer_question_uses_standard_search(monkeypatch):
    places = [object()]

    monkeypatch.setattr("src.rag.answer.search_places", lambda question, limit: places)
    monkeypatch.setattr(
        "src.rag.answer._answer_question",
        lambda question, result_places: RagAnswer(answer=f"Answer for {question}"),
    )

    result = answer_question("Where is the lake?", limit=2)

    assert result.agentic is False
    assert result.places is places
    assert result.attempts is None
    assert result.sufficient is None
    assert result.response.answer == "Answer for Where is the lake?"


def test_answer_question_uses_agentic_search(monkeypatch):
    agentic_result = type(
        "AgenticResult",
        (),
        {
            "places": [object(), object()],
            "attempts": ["attempt-1"],
            "sufficient": True,
        },
    )()

    monkeypatch.setattr(
        "src.rag.answer.agentic_search_places",
        lambda question, limit: agentic_result,
    )
    monkeypatch.setattr(
        "src.rag.answer._answer_question",
        lambda question, result_places: RagAnswer(answer="Agentic answer"),
    )

    result = answer_question("Need more context", limit=4, agentic=True)

    assert result.agentic is True
    assert result.places == agentic_result.places
    assert result.attempts == agentic_result.attempts
    assert result.sufficient is True
    assert result.response.answer == "Agentic answer"