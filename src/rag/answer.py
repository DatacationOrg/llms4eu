import argparse
from pathlib import Path

from pydantic import BaseModel, Field

from src.rag.search import ScoredPlace, search_places
from src.shared.env import load_local_env, load_yaml
from src.shared.llm import structured_local_model


__all__ = ["ANSWER_CONFIG", "RagAnswer", "answer", "prompt_for"]

ANSWER_CONFIG = load_yaml(Path(__file__).with_name("answer_config.yaml"))


class RagAnswer(BaseModel):
    answer: str = Field(description="A short answer based only on the provided context.")


def answer() -> None:
    """CLI entrypoint for retrieval plus a local structured LLM answer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--limit", type=int, default=ANSWER_CONFIG["default_limit"])
    args = parser.parse_args()

    load_local_env()
    places = search_places(args.question, args.limit)
    _print_retrieved(places)
    response = _answer_question(args.question, places)

    print("\nAnswer:")
    print(response.answer)


def prompt_for(question: str, places: list[ScoredPlace]) -> str:
    """Build the exact prompt printed by the answer script."""
    context = "\n\n".join(place.context_text for place in places)
    return (
        "Answer the question using only this context. "
        "If the context is not enough, say so.\n\n"
        f"context:\n{context}\n\nquestion: {question}"
    )


# ---------- PRIVATE FUNCTIONS ----------


def _print_retrieved(places: list[ScoredPlace]) -> None:
    # These logs make it obvious whether a bad answer came from retrieval or LLM.
    print("Retrieved:")
    for place in places:
        print(f"- {place.score:.3f} {place.id}: {place.summary}")


def _answer_question(question: str, places: list[ScoredPlace]) -> RagAnswer:
    # Keep prompt construction explicit so retrieved context is easy to inspect.
    prompt = prompt_for(question, places)
    print("\nContext:")
    print(prompt)

    return structured_local_model(ANSWER_CONFIG["answer_model"], RagAnswer).invoke(prompt)


if __name__ == "__main__":
    answer()
