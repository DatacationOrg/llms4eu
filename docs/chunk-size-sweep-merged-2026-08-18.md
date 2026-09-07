# Chunk Size Sweep — Merged 2026-08-14 + 2026-08-17 + 2026-08-18

One table per metric across 12 retrieval methods and
5 chunk variants, joined from 3 runs. Derived:
regenerate with `experiments/indexing/merge_chunk_sweeps.py` rather than
editing by hand.

| source | methods | measured |
|---|---|---|
| [`chunk-size-sweep-2026-08-14.md`](chunk-size-sweep-2026-08-14.md) | `qwen`, `qwen_hybrid_rerank`, `nemotron` | 2026-08-14 |
| [`chunk-size-sweep-2026-08-17.md`](chunk-size-sweep-2026-08-17.md) | `nemotron_hybrid`, `nemotron_hybrid_rerank` | 2026-08-17 |
| [`chunk-size-sweep-2026-08-18-qwen4b.md`](chunk-size-sweep-2026-08-18-qwen4b.md), [`chunk-size-sweep-2026-08-18-qwen8b.md`](chunk-size-sweep-2026-08-18-qwen8b.md), [`chunk-size-sweep-2026-08-18-nemotron8b.md`](chunk-size-sweep-2026-08-18-nemotron8b.md), [`chunk-size-sweep-2026-08-18-rerank4b.md`](chunk-size-sweep-2026-08-18-rerank4b.md) | `qwen4b`, `qwen8b`, `nemotron8b`, `qwen4b_hybrid_rerank`, `qwen8b_hybrid_rerank`, `nemotron8b_hybrid_rerank`, `qwen8b_hybrid_rerank_4b` | 2026-08-18 |

## Why these runs can be joined

Every run used the **per-variant** question design over the same five cuttings
in `.local/db/pages-variants.db`, the same vector store, and the same
post-2026-08-11 sequence lengths, so no column is truncated. Questions scored
per variant, identical in all runs:

| variant | chunks | questions scored |
|---|---:|---:|
| base | 726 | 3471 |
| tok256 | 2003 | 7801 |
| tok512 | 990 | 3605 |
| tok512ov | 1049 | 3865 |
| tok1024 | 501 | 1451 |

Verified equal across every run, so a row is not comparing two samples.

## What each run added

**08-14** measured three methods, but gave only `qwen_hybrid_rerank` the
hybrid+rerank pipeline; `nemotron` ran as bare dense retrieval. Those cells
differ by 260x in cost (11.3s against 2,922.5s on `base`), so that table
compared pipelines where it meant to compare chunkings.

**08-17** added the two rungs nemotron was missing, on the same corpus, so
nemotron's row shows its capability rather than its floor.

**08-18** added scale. Both earlier runs ranked chunkings using only the two
smallest embedders on the machine — Qwen3-Embedding-0.6B and
Nemotron-3-Embed-1B — so every chunking conclusion rested on models small
enough that they might not resolve a difference at all. This run adds
Qwen3-Embedding-4B and 8B, Nemotron-3-Embed-8B, and one larger *reranker*
rung (`qwen8b_hybrid_rerank_4b`, Qwen3-Reranker-4B in place of the 0.6B),
because the finding both earlier runs agreed on is that the reranker dominates
and the reranker doing that work was the smallest in its family.

Read **down** a column for the pipeline and model-size effect, **across** a
row for the chunking effect. Bold is the best cell in that metric.

## hit@1

| method | run | base | tok256 | tok512 | tok512ov | tok1024 |
|---|---|---:|---:|---:|---:|---:|
| `qwen` | 08-14 | 0.518 | 0.569 | 0.581 | 0.558 | 0.612 |
| `qwen4b` | 08-18 | 0.657 | 0.695 | 0.709 | 0.693 | 0.731 |
| `qwen8b` | 08-18 | 0.691 | 0.730 | 0.731 | 0.718 | 0.752 |
| `nemotron` | 08-14 | 0.458 | 0.444 | 0.482 | 0.454 | 0.517 |
| `nemotron8b` | 08-18 | 0.802 | 0.826 | 0.840 | 0.821 | 0.829 |
| `nemotron_hybrid` | 08-17 | 0.546 | 0.518 | 0.573 | 0.538 | 0.604 |
| `qwen_hybrid_rerank` | 08-14 | 0.749 | 0.739 | 0.781 | 0.754 | 0.826 |
| `qwen4b_hybrid_rerank` | 08-18 | 0.764 | 0.750 | 0.792 | 0.765 | 0.837 |
| `qwen8b_hybrid_rerank` | 08-18 | 0.764 | 0.751 | 0.789 | 0.766 | 0.837 |
| `qwen8b_hybrid_rerank_4b` | 08-18 | 0.845 | - | - | - | **0.898** |
| `nemotron_hybrid_rerank` | 08-17 | 0.746 | 0.720 | 0.765 | 0.743 | 0.811 |
| `nemotron8b_hybrid_rerank` | 08-18 | 0.765 | 0.754 | 0.793 | 0.765 | 0.837 |

## hit@5

| method | run | base | tok256 | tok512 | tok512ov | tok1024 |
|---|---|---:|---:|---:|---:|---:|
| `qwen` | 08-14 | 0.737 | 0.777 | 0.817 | 0.802 | 0.841 |
| `qwen4b` | 08-18 | 0.848 | 0.877 | 0.909 | 0.896 | 0.922 |
| `qwen8b` | 08-18 | 0.884 | 0.898 | 0.931 | 0.916 | 0.945 |
| `nemotron` | 08-14 | 0.664 | 0.645 | 0.715 | 0.693 | 0.733 |
| `nemotron8b` | 08-18 | 0.940 | 0.943 | 0.969 | 0.960 | 0.970 |
| `nemotron_hybrid` | 08-17 | 0.744 | 0.713 | 0.787 | 0.774 | 0.824 |
| `qwen_hybrid_rerank` | 08-14 | 0.880 | 0.881 | 0.931 | 0.914 | 0.948 |
| `qwen4b_hybrid_rerank` | 08-18 | 0.907 | 0.903 | 0.946 | 0.935 | 0.966 |
| `qwen8b_hybrid_rerank` | 08-18 | 0.912 | 0.906 | 0.948 | 0.933 | 0.970 |
| `qwen8b_hybrid_rerank_4b` | 08-18 | 0.950 | - | - | - | **0.986** |
| `nemotron_hybrid_rerank` | 08-17 | 0.875 | 0.848 | 0.902 | 0.898 | 0.925 |
| `nemotron8b_hybrid_rerank` | 08-18 | 0.917 | 0.909 | 0.950 | 0.939 | 0.968 |

## hit@10

| method | run | base | tok256 | tok512 | tok512ov | tok1024 |
|---|---|---:|---:|---:|---:|---:|
| `qwen` | 08-14 | 0.797 | 0.828 | 0.877 | 0.855 | 0.897 |
| `qwen4b` | 08-18 | 0.899 | 0.910 | 0.947 | 0.929 | 0.957 |
| `qwen8b` | 08-18 | 0.928 | 0.931 | 0.962 | 0.949 | 0.972 |
| `nemotron` | 08-14 | 0.727 | 0.708 | 0.783 | 0.767 | 0.813 |
| `nemotron8b` | 08-18 | 0.964 | 0.961 | 0.983 | 0.977 | 0.989 |
| `nemotron_hybrid` | 08-17 | 0.824 | 0.800 | 0.865 | 0.853 | 0.888 |
| `qwen_hybrid_rerank` | 08-14 | 0.908 | 0.905 | 0.954 | 0.937 | 0.968 |
| `qwen4b_hybrid_rerank` | 08-18 | 0.936 | 0.931 | 0.973 | 0.960 | 0.983 |
| `qwen8b_hybrid_rerank` | 08-18 | 0.945 | 0.937 | 0.975 | 0.965 | 0.988 |
| `qwen8b_hybrid_rerank_4b` | 08-18 | 0.961 | - | - | - | **0.992** |
| `nemotron_hybrid_rerank` | 08-17 | 0.893 | 0.868 | 0.920 | 0.916 | 0.940 |
| `nemotron8b_hybrid_rerank` | 08-18 | 0.954 | 0.942 | 0.979 | 0.969 | 0.989 |

## recall@10

| method | run | base | tok256 | tok512 | tok512ov | tok1024 |
|---|---|---:|---:|---:|---:|---:|
| `qwen` | 08-14 | 0.797 | 0.828 | 0.877 | 0.855 | 0.897 |
| `qwen4b` | 08-18 | 0.899 | 0.910 | 0.947 | 0.929 | 0.957 |
| `qwen8b` | 08-18 | 0.928 | 0.931 | 0.962 | 0.949 | 0.972 |
| `nemotron` | 08-14 | 0.727 | 0.708 | 0.783 | 0.767 | 0.813 |
| `nemotron8b` | 08-18 | 0.964 | 0.961 | 0.983 | 0.977 | 0.989 |
| `nemotron_hybrid` | 08-17 | 0.824 | 0.800 | 0.865 | 0.853 | 0.888 |
| `qwen_hybrid_rerank` | 08-14 | 0.908 | 0.905 | 0.954 | 0.937 | 0.968 |
| `qwen4b_hybrid_rerank` | 08-18 | 0.936 | 0.931 | 0.973 | 0.960 | 0.983 |
| `qwen8b_hybrid_rerank` | 08-18 | 0.945 | 0.937 | 0.975 | 0.965 | 0.988 |
| `qwen8b_hybrid_rerank_4b` | 08-18 | 0.961 | - | - | - | **0.992** |
| `nemotron_hybrid_rerank` | 08-17 | 0.893 | 0.868 | 0.920 | 0.916 | 0.940 |
| `nemotron8b_hybrid_rerank` | 08-18 | 0.954 | 0.942 | 0.979 | 0.969 | 0.989 |

## mrr@10

| method | run | base | tok256 | tok512 | tok512ov | tok1024 |
|---|---|---:|---:|---:|---:|---:|
| `qwen` | 08-14 | 0.611 | 0.658 | 0.683 | 0.665 | 0.710 |
| `qwen4b` | 08-18 | 0.742 | 0.774 | 0.796 | 0.783 | 0.815 |
| `qwen8b` | 08-18 | 0.775 | 0.803 | 0.817 | 0.805 | 0.833 |
| `nemotron` | 08-14 | 0.547 | 0.531 | 0.584 | 0.559 | 0.613 |
| `nemotron8b` | 08-18 | 0.863 | 0.876 | 0.897 | 0.883 | 0.890 |
| `nemotron_hybrid` | 08-17 | 0.631 | 0.606 | 0.668 | 0.642 | 0.698 |
| `qwen_hybrid_rerank` | 08-14 | 0.808 | 0.800 | 0.847 | 0.826 | 0.880 |
| `qwen4b_hybrid_rerank` | 08-18 | 0.827 | 0.815 | 0.860 | 0.840 | 0.892 |
| `qwen8b_hybrid_rerank` | 08-18 | 0.829 | 0.818 | 0.859 | 0.841 | 0.893 |
| `qwen8b_hybrid_rerank_4b` | 08-18 | 0.891 | - | - | - | **0.937** |
| `nemotron_hybrid_rerank` | 08-17 | 0.803 | 0.776 | 0.825 | 0.813 | 0.860 |
| `nemotron8b_hybrid_rerank` | 08-18 | 0.832 | 0.822 | 0.861 | 0.844 | 0.893 |

## Speed

Milliseconds per query. The 08-14 figures are derived from its Coverage table
(total seconds / questions scored); later runs measure per query.

| method | run | base | tok256 | tok512 | tok512ov | tok1024 |
|---|---|---:|---:|---:|---:|---:|
| `qwen` | 08-14 | 3.3 | 2.7 | 2.4 | 2.2 | 2.1 |
| `qwen4b` | 08-18 | 19.0 | 8.0 | 7.5 | 6.9 | 7.0 |
| `qwen8b` | 08-18 | 17.1 | 12.9 | 15.3 | 20.1 | 10.9 |
| `nemotron` | 08-14 | 3.3 | 2.5 | 2.1 | 2.1 | 2.0 |
| `nemotron8b` | 08-18 | 11.2 | 8.4 | 7.7 | 7.2 | 7.1 |
| `nemotron_hybrid` | 08-17 | 2.0 | 2.9 | 1.9 | 2.3 | 1.5 |
| `qwen_hybrid_rerank` | 08-14 | 842.0 | 292.0 | 500.2 | 502.5 | 912.6 |
| `qwen4b_hybrid_rerank` | 08-18 | 1133.6 | 287.4 | 496.0 | 498.4 | 902.2 |
| `qwen8b_hybrid_rerank` | 08-18 | 692.1 | 463.2 | 929.9 | 676.2 | 903.8 |
| `qwen8b_hybrid_rerank_4b` | 08-18 | 2713.7 | - | - | - | 3562.9 |
| `nemotron_hybrid_rerank` | 08-17 | 713.9 | 307.1 | 549.6 | 545.9 | 1006.2 |
| `nemotron8b_hybrid_rerank` | 08-18 | 702.7 | 290.2 | 501.0 | 503.2 | 917.4 |

## hit@5 by category

Only runs that published a category split appear here. This is the part of
the data that disagrees with the aggregate.

| method | variant | crosslingual | direct_long | direct_short | vague_long | vague_short |
|---|---|---:|---:|---:|---:|---:|
| `qwen4b` | base | 0.866 | 0.971 | 0.869 | 0.909 | 0.637 |
| `qwen4b` | tok256 | 0.840 | 0.955 | 0.929 | 0.859 | 0.785 |
| `qwen4b` | tok512 | 0.879 | 0.969 | 0.934 | 0.895 | 0.852 |
| `qwen4b` | tok512ov | 0.859 | 0.964 | 0.929 | 0.895 | 0.816 |
| `qwen4b` | tok1024 | 0.914 | 0.983 | 0.957 | 0.913 | 0.842 |
| `qwen8b` | base | 0.896 | 0.976 | 0.902 | 0.944 | 0.715 |
| `qwen8b` | tok256 | 0.874 | 0.966 | 0.938 | 0.878 | 0.824 |
| `qwen8b` | tok512 | 0.897 | 0.980 | 0.950 | 0.921 | 0.889 |
| `qwen8b` | tok512ov | 0.885 | 0.979 | 0.940 | 0.911 | 0.853 |
| `qwen8b` | tok1024 | 0.951 | 0.990 | 0.970 | 0.940 | 0.875 |
| `nemotron8b` | base | 0.954 | 0.989 | 0.944 | 0.988 | 0.833 |
| `nemotron8b` | tok256 | 0.909 | 0.983 | 0.971 | 0.937 | 0.899 |
| `nemotron8b` | tok512 | 0.958 | 0.994 | 0.979 | 0.966 | 0.942 |
| `nemotron8b` | tok512ov | 0.931 | 0.994 | 0.973 | 0.965 | 0.924 |
| `nemotron8b` | tok1024 | 0.971 | 0.997 | 0.974 | 0.973 | 0.934 |
| `nemotron_hybrid` | base | 0.793 | 0.899 | 0.766 | 0.825 | 0.453 |
| `nemotron_hybrid` | tok256 | 0.696 | 0.867 | 0.786 | 0.627 | 0.578 |
| `nemotron_hybrid` | tok512 | 0.734 | 0.907 | 0.891 | 0.711 | 0.663 |
| `nemotron_hybrid` | tok512ov | 0.749 | 0.915 | 0.849 | 0.692 | 0.650 |
| `nemotron_hybrid` | tok1024 | 0.823 | 0.924 | 0.894 | 0.749 | 0.730 |
| `qwen4b_hybrid_rerank` | base | 0.890 | 0.989 | 0.906 | 0.967 | 0.795 |
| `qwen4b_hybrid_rerank` | tok256 | 0.816 | 0.970 | 0.939 | 0.888 | 0.857 |
| `qwen4b_hybrid_rerank` | tok512 | 0.881 | 0.983 | 0.970 | 0.947 | 0.919 |
| `qwen4b_hybrid_rerank` | tok512ov | 0.840 | 0.988 | 0.969 | 0.936 | 0.899 |
| `qwen4b_hybrid_rerank` | tok1024 | 0.938 | 0.987 | 0.990 | 0.963 | 0.947 |
| `qwen8b_hybrid_rerank` | base | 0.890 | 0.986 | 0.912 | 0.968 | 0.812 |
| `qwen8b_hybrid_rerank` | tok256 | 0.823 | 0.971 | 0.937 | 0.895 | 0.861 |
| `qwen8b_hybrid_rerank` | tok512 | 0.874 | 0.984 | 0.968 | 0.952 | 0.927 |
| `qwen8b_hybrid_rerank` | tok512ov | 0.844 | 0.987 | 0.969 | 0.936 | 0.889 |
| `qwen8b_hybrid_rerank` | tok1024 | 0.938 | 0.987 | 0.993 | 0.970 | 0.957 |
| `qwen8b_hybrid_rerank_4b` | base | 0.959 | 0.992 | 0.949 | 0.988 | 0.867 |
| `qwen8b_hybrid_rerank_4b` | tok1024 | 0.979 | 0.993 | 0.993 | 0.990 | 0.970 |
| `nemotron_hybrid_rerank` | base | 0.858 | 0.973 | 0.893 | 0.929 | 0.734 |
| `nemotron_hybrid_rerank` | tok256 | 0.760 | 0.952 | 0.924 | 0.789 | 0.771 |
| `nemotron_hybrid_rerank` | tok512 | 0.816 | 0.974 | 0.964 | 0.875 | 0.837 |
| `nemotron_hybrid_rerank` | tok512ov | 0.805 | 0.978 | 0.963 | 0.861 | 0.839 |
| `nemotron_hybrid_rerank` | tok1024 | 0.885 | 0.970 | 0.987 | 0.906 | 0.868 |
| `nemotron8b_hybrid_rerank` | base | 0.900 | 0.992 | 0.915 | 0.973 | 0.815 |
| `nemotron8b_hybrid_rerank` | tok256 | 0.825 | 0.974 | 0.939 | 0.895 | 0.871 |
| `nemotron8b_hybrid_rerank` | tok512 | 0.881 | 0.984 | 0.966 | 0.952 | 0.935 |
| `nemotron8b_hybrid_rerank` | tok512ov | 0.846 | 0.988 | 0.969 | 0.936 | 0.912 |
| `nemotron8b_hybrid_rerank` | tok1024 | 0.930 | 0.990 | 0.987 | 0.970 | 0.954 |

## Read this before quoting a winner

**No span metric is available in any of these runs.** All used per-variant
question sets, and the `gold` span target is read from `base` chunks
(`src/eval/evaluate.py:load_span_labels`), which those questions have no link
to — so `Anchored` is 0 in every cell and the character-overlap sections are
empty. Everything here is ranked on `hit@k`/`recall@k`, which count whole
chunks and therefore reward a variant for cutting large, independently of how
well it was cut.

**A variant with more chunks is also a harder haystack**, so `tok256` searches
2003 chunks where `tok1024` searches
501. Read a large gap as real and a small one as
possibly just haystack size.

To rank these variants defensibly, anchor the answers
(`just anchor-answers`), project the labels (`just relabel <variant>`), and
re-run under `just shared-overview`, where every column scores one shared
question set and the span metrics resolve.

**Model size is confounded with nothing here, but language coverage is.**
Nemotron-3-Embed's model card declares 42 languages and Slovenian is not one
of them, at either 1B or 8B. This corpus is Slovenian. A nemotron-to-qwen gap
is therefore not evidence about model scale, and the 1B-to-8B step is the row
that separates the two readings.

