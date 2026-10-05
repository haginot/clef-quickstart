# Clef / Clef-Flash / Jev comparison

The same item IDs, gold labels, task wording, and one-request-per-item protocol were used for all three systems. Binary tasks use AUROC; ANLI R3 uses accuracy.

| task | decisions | Jev | Clef | Clef-Flash |
|---|---:|---:|---:|---:|
| ToxicChat | 600 | **0.990** | 0.986 | 0.977 |
| BeaverTails | 500 | 0.900 | **0.905** | 0.874 |
| HaluEval QA | 600 | **0.940** | 0.856 | 0.329 |
| HaluEval Dialogue | 600 | **0.914** | 0.896 | 0.807 |
| BoolQ | 500 | **0.977** | 0.975 | 0.970 |
| ANLI R3 | 500 | **0.662** | 0.534 | 0.518 |
| Prompt Injection | 500 | **0.994** | 0.988 | 0.988 |
| Macro mean | 3,800 | **0.911** | 0.877 | 0.781 |

## Hosted API observations

| system | total cost (3,800 decisions) | cost per 1,000 | task p50 range |
|---|---:|---:|---:|
| Jev | $0.0657 | $0.0173 | 258–270 ms |
| Clef-Flash | $0.0911 | $0.0240 | 237–310 ms |
| Clef | $0.2429 | $0.0639 | 545–666 ms |

The Jev requests were made in September 2026. The Clef requests were made on October 6, 2026 through OpenRouter. These client-observed times are not controlled, same-time model latency measurements. Requests ran concurrently, so they also must not be interpreted as single-worker throughput.

## HaluEval QA polarity check

The low Clef-Flash AUROC is not caused by an ID or gold-label mismatch. Mean predicted probability for “hallucinated” was:

| model | correct answer (gold 0) | hallucinated answer (gold 1) |
|---|---:|---:|
| Clef | 0.120 | 0.486 |
| Clef-Flash | 0.727 | 0.569 |

Clef-Flash ranked the correct answers as more hallucinated on average, which explains the below-random AUROC on this task.
