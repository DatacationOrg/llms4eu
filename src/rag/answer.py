import argparse
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from src.rag.agent_search import agentic_search_places
from src.rag.search import ScoredPlace, search_places
from src.shared.env import load_local_env, load_yaml
from src.shared.llm import structured_local_model


__all__ = [
    "ANSWER_CONFIG",
    "AnswerResult",
    "RagAnswer",
    "answer",
    "answer_question",
    "prompt_for",
]

ANSWER_CONFIG = load_yaml(Path(__file__).with_name("config.yaml"))["answer"]


class RagAnswer(BaseModel):
    answer: str = Field(
        description="A short answer based only on the provided context."
    )


@dataclass(frozen=True)
class AnswerResult:
    response: RagAnswer
    places: list[ScoredPlace]
    agentic: bool
    sufficient: bool | None = None
    attempts: list | None = None


def answer() -> None:
    """CLI entrypoint for retrieval plus a local structured LLM answer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--limit", type=int, default=ANSWER_CONFIG["default_limit"])
    parser.add_argument(
        "--agentic",
        action="store_true",
        help="Retry retrieval until context is sufficient or attempt budget is reached.",
    )
    args = parser.parse_args()

    load_local_env()
    result = answer_question(
        args.question,
        limit=args.limit,
        agentic=args.agentic,
    )
    if result.attempts:
        _print_attempts(result.attempts)
    _print_retrieved(result.places)

    print("\nAnswer:")
    print(result.response.answer)


def prompt_for(question: str, places: list[ScoredPlace]) -> str:
    """Build the exact prompt printed by the answer script."""
    context = "\n\n".join(place.context_text for place in places)
    return (
        "Answer the question using only this context. "
        "If the context is not enough, say so.\n\n"
        f"context:\n{context}\n\nquestion: {question}"
    )


def answer_question(
    question: str,
    *,
    limit: int | None = None,
    agentic: bool = False,
) -> AnswerResult:
    if agentic:
        result = agentic_search_places(question, limit)
        places = result.places
        attempts = result.attempts
        sufficient = result.sufficient
    else:
        places = search_places(question, limit)
        attempts = None
        sufficient = None
    response = _answer_question(question, places)
    return AnswerResult(
        response=response,
        places=places,
        agentic=agentic,
        sufficient=sufficient,
        attempts=attempts,
    )


# ---------- PRIVATE FUNCTIONS ----------


def _print_retrieved(places: list[ScoredPlace]) -> None:
    # These logs make it obvious whether a bad answer came from retrieval or LLM.
    print("Retrieved:")
    for place in places:
        print(f"- {place.score:.3f} {place.id}: {place.summary}")


def _print_attempts(attempts: list) -> None:
    print("Agentic search attempts:")
    for attempt in attempts:
        print(
            "- "
            f"#{attempt.state.attempt} "
            f"strategy={attempt.state.strategy} "
            f"model={attempt.state.embedding_model} "
            f"limit={attempt.state.limit} "
            f"hits={attempt.hits} "
            f"sufficient={attempt.sufficient} "
            f"reason={attempt.reason}"
        )


def _answer_question(question: str, places: list[ScoredPlace]) -> RagAnswer:
    # Keep prompt construction explicit so retrieved context is easy to inspect.
    prompt = prompt_for(question, places)
    print("\nContext:")
    print(prompt)

    return structured_local_model(ANSWER_CONFIG["model"], RagAnswer).invoke(prompt)


if __name__ == "__main__":
    answer()
