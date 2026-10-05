# Clef / Clef-Flash comparison run

- Date: 2026-10-06 (Asia/Tokyo)
- Goal: compare Cloudflare Clef and Clef-Flash with the saved Jev results used in the earlier article.
- Hosted API: OpenRouter `POST /api/v1/systemone`
- Models: `cloudflare/clef`, `cloudflare/clef-flash`
- Protocol: one item and one typed question per API request; no batching.
- Data: the same seeded public benchmark slices and task wording as `haginot/jev-graph-rag-qa`.
- Primary metric: AUROC for six binary tasks; accuracy for ANLI R3.
- Calibration: Brier score and 10-bin equal-width ECE.
- Latency: client-observed wall time per request, with concurrent workers; interpret as an end-to-end observation, not isolated model compute time.
- Raw predictions: written below `outputs/benchmark/` and intentionally ignored by Git.
- Public summaries: written below `experiments/results/`.

## Sample counts

| task | decisions |
|---|---:|
| ToxicChat | 600 |
| BeaverTails | 500 |
| HaluEval QA | 600 |
| HaluEval Dialogue | 600 |
| BoolQ | 500 |
| ANLI R3 | 500 |
| Prompt Injection | 500 |
| Total | 3,800 |

## Reproduction

```bash
uv run --with-requirements requirements-experiments.txt \
  python experiments/run_benchmark.py \
  --model cloudflare/clef-flash \
  --jev-archive /path/to/jev_bench_predictions_2026-09-21.tar.gz
```

Set `OPENROUTER_API_KEY` outside the repository before running. Repeat with
`--model cloudflare/clef`. Interrupted runs resume from their JSONL output.
