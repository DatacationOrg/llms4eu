from src.eval.equivalence import (
    EvidenceDocument,
    EvidenceEquivalence,
    EvidenceEquivalenceJudge,
)


class FakeStructuredLlm:
    def __init__(self, result: EvidenceEquivalence) -> None:
        self.result = result
        self.prompt = ""

    def structured_output(self, prompt, output_schema, *, retries=3):
        self.prompt = prompt
        assert output_schema is EvidenceEquivalence
        assert retries == 2
        return self.result


def test_equivalence_judge_compares_question_specific_evidence():
    client = FakeStructuredLlm(
        EvidenceEquivalence(
            verdict="equivalent",
            near_duplicate_ids=["retrieved-1", "invented", "retrieved-1"],
            confidence=0.95,
            reason="Both chunks state the same opening date.",
        )
    )
    judge = EvidenceEquivalenceJudge(client=client, retries=2)

    result = judge.evaluate(
        question="When did the museum open?",
        expected_answer="It opened in 1998.",
        golden=[EvidenceDocument("gold", "The museum opened in 1998.")],
        retrieved=[EvidenceDocument("retrieved-1", "Its opening took place in 1998.")],
    )

    assert result.verdict == "equivalent"
    assert result.near_duplicate_ids == ["retrieved-1"]
    assert "Shared entities, keywords, or general topic are not enough" in client.prompt
    assert '<document id="gold">' in client.prompt
    assert '<document id="retrieved-1">' in client.prompt


def test_equivalence_judge_handles_empty_retrieval_without_model_call():
    client = FakeStructuredLlm(
        EvidenceEquivalence(
            verdict="equivalent",
            confidence=1,
            reason="unused",
        )
    )
    judge = EvidenceEquivalenceJudge(client=client)

    result = judge.evaluate(
        question="Question",
        expected_answer="Answer",
        golden=[EvidenceDocument("gold", "Evidence")],
        retrieved=[],
    )

    assert result.verdict == "different"
    assert result.confidence == 1
    assert client.prompt == ""
