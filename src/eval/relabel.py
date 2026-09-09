"""Point the generated eval questions at a new chunk variant's chunks.

The questions are the fixed part of the benchmark: their text, answers and
categories never change, which is what lets one retrieval method's score be
compared straight across variants. What a chunking ablation changes is *which
chunk* holds the answer, because every variant cuts the pages differently. This
regenerates only that link.

Two stages, and only the first involves a model:

1. **Anchor** each answer to a verbatim quote's character span in its page
   (`src.eval.anchors`). Runs once for the whole benchmark, is checkable, and is
   independent of any chunking.
2. **Project** that span onto the variant's chunks by interval overlap. Pure
   arithmetic, so it adds no noise to the numbers being compared and returns the
   same answer every time.

The rejected alternative was asking a model, per variant, which chunk holds the
answer. That repeats an unverifiable judgement for every variant and puts its
noise straight into the comparison — asymmetrically, because `base`'s own labels
were never judged: its questions were generated *from* their gold chunk, so that
chunk is gold by construction. Anchoring puts every variant, `base` included, on
the same footing.

Generating fresh questions per variant was rejected for a worse reason. A
question written from the chunk it will then be used to retrieve gets easier the
smaller that chunk is, so small chunks would win by construction. That is bias,
not noise, and it does not average out.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from src.eval.anchors import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_WORKERS,
    extract_anchors,
    label_variant,
)
from src.shared.llm import StructuredLlm

__all__ = ["RelabelSummary", "relabel_variant"]


@dataclass(frozen=True)
class RelabelSummary:
    variant: str
    anchored: int
    questions: int
    links: int
    rejected: int = 0
    failed: int = 0

    def __str__(self) -> str:
        detail = ""
        if self.rejected or self.failed:
            detail = (
                f" (anchoring: {self.rejected} rejected, {self.failed} model failures)"
            )
        return (
            f"{self.variant}: {self.questions} questions linked to {self.links} "
            f"chunks, {self.anchored} new anchors{detail}"
        )


def relabel_variant(
    variant: str,
    limit: int | None = None,
    llm: StructuredLlm | None = None,
    workers: int = DEFAULT_WORKERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> RelabelSummary:
    """Anchor any unanchored answers, then link them to `variant`'s chunks.

    Questions that already have an anchor are skipped, so the model pass happens
    once for the whole benchmark however many variants are added afterwards.
    """
    anchors = extract_anchors(
        limit=limit, llm=llm, workers=workers, batch_size=batch_size
    )
    questions, links = label_variant(variant)
    return RelabelSummary(
        variant=variant,
        anchored=anchors.anchored,
        questions=questions,
        links=links,
        rejected=anchors.rejected,
        failed=anchors.failed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    print(
        relabel_variant(
            args.variant,
            limit=args.limit,
            workers=args.workers,
            batch_size=args.batch_size,
        )
    )


if __name__ == "__main__":
    main()
