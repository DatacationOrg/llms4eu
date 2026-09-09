# Chunk Token Audit 2026-08-11

## Run Configuration

- Command: `experiments/indexing/audit_chunk_tokens.py --providers qwen,qwen4b --versions v1,v2 --output docs/reports/chunking/chunk-token-audit-2026-08-11.md`
- SQLite chunks scanned: 726
- Representations: v1, v2
- Token accounting: document prompt + representation text + special tokens, with truncation disabled
- Chunk target/max/min characters: 1800/2600/300
- Models: qwen=Qwen/Qwen3-Embedding-0.6B (configured 512, effective 512); qwen4b=Qwen/Qwen3-Embedding-4B (configured 512, effective 512)

## Corpus Results

| Provider | Version | Language | N | Chars p50/p90/p95/p99/max | Tokens p50/p90/p95/p99/max | Tok/char | Over limit | Lost total/max | Overhead mean/max | Worst IDs |
|---|---|---|---:|---|---|---:|---:|---:|---:|---|
| qwen | v1 | sl | 726 | 1572.5/2324/2544/2593.75/2599 | 616.5/921.5/1055.5/1179.25/1253 | 0.4081 | 506 (69.70%) | 103114/741 | 24.3/56 | f300c213-5cd2-5282-94b1-288c2d8e4a3f:0, 7eee812e-67ef-5d70-b2b1-6377671d06bc:0, f1310a30-23d0-5e22-91bf-ceb08abba33a:20, f1310a30-23d0-5e22-91bf-ceb08abba33a:21, 3fe2cd5a-1e83-5f45-8f36-e7881bb51aa5:0 |
| qwen | v2 | sl | 726 | 1572.5/2324/2544/2593.75/2599 | 635/941.5/1074/1200.5/1275 | 0.4218 | 524 (72.18%) | 113250/763 | 44.4/81 | f300c213-5cd2-5282-94b1-288c2d8e4a3f:0, 7eee812e-67ef-5d70-b2b1-6377671d06bc:0, f1310a30-23d0-5e22-91bf-ceb08abba33a:20, f1310a30-23d0-5e22-91bf-ceb08abba33a:21, 3fe2cd5a-1e83-5f45-8f36-e7881bb51aa5:0 |
| qwen4b | v1 | sl | 726 | 1572.5/2324/2544/2593.75/2599 | 616.5/921.5/1055.5/1179.25/1253 | 0.4081 | 506 (69.70%) | 103114/741 | 24.3/56 | f300c213-5cd2-5282-94b1-288c2d8e4a3f:0, 7eee812e-67ef-5d70-b2b1-6377671d06bc:0, f1310a30-23d0-5e22-91bf-ceb08abba33a:20, f1310a30-23d0-5e22-91bf-ceb08abba33a:21, 3fe2cd5a-1e83-5f45-8f36-e7881bb51aa5:0 |
| qwen4b | v2 | sl | 726 | 1572.5/2324/2544/2593.75/2599 | 635/941.5/1074/1200.5/1275 | 0.4218 | 524 (72.18%) | 113250/763 | 44.4/81 | f300c213-5cd2-5282-94b1-288c2d8e4a3f:0, 7eee812e-67ef-5d70-b2b1-6377671d06bc:0, f1310a30-23d0-5e22-91bf-ceb08abba33a:20, f1310a30-23d0-5e22-91bf-ceb08abba33a:21, 3fe2cd5a-1e83-5f45-8f36-e7881bb51aa5:0 |

Character percentiles use canonical chunk text. Token percentiles use the effective embedding input. Overhead includes representation metadata, the model's document prompt, and special tokens.

## Worst Inputs

| Provider | Version | Chunk ID | Tokens | Limit | Lost | Overhead |
|---|---|---|---:|---:|---:|---:|
| qwen | v2 | f300c213-5cd2-5282-94b1-288c2d8e4a3f:0 | 1275 | 512 | 763 | 61 |
| qwen4b | v2 | f300c213-5cd2-5282-94b1-288c2d8e4a3f:0 | 1275 | 512 | 763 | 61 |
| qwen | v2 | 7eee812e-67ef-5d70-b2b1-6377671d06bc:0 | 1258 | 512 | 746 | 63 |
| qwen4b | v2 | 7eee812e-67ef-5d70-b2b1-6377671d06bc:0 | 1258 | 512 | 746 | 63 |
| qwen | v1 | f300c213-5cd2-5282-94b1-288c2d8e4a3f:0 | 1253 | 512 | 741 | 39 |
| qwen4b | v1 | f300c213-5cd2-5282-94b1-288c2d8e4a3f:0 | 1253 | 512 | 741 | 39 |
| qwen | v2 | f1310a30-23d0-5e22-91bf-ceb08abba33a:20 | 1240 | 512 | 728 | 37 |
| qwen4b | v2 | f1310a30-23d0-5e22-91bf-ceb08abba33a:20 | 1240 | 512 | 728 | 37 |
| qwen | v1 | 7eee812e-67ef-5d70-b2b1-6377671d06bc:0 | 1236 | 512 | 724 | 41 |
| qwen4b | v1 | 7eee812e-67ef-5d70-b2b1-6377671d06bc:0 | 1236 | 512 | 724 | 41 |

## Slovenian/English Pairs

Blocked pending a trustworthy 30-pair reviewed translation fixture. No review status was inferred or fabricated. Detail: Reviewed language-pair fixture is unavailable: /home/gerson/llms4eu/experiments/indexing/data/sl_en_tourism_pairs.jsonl

## Recommendation

Do not change stable chunk IDs from this audit alone: 2060/2904 inputs (70.94%) exceed a provider limit. Run a future versioned, provider-aware token-budget experiment unless a lower shared character target is shown to satisfy every provider.
