# cloudflare/clef-flash benchmark result

One request per item. Binary tasks use AUROC as the primary metric; ANLI R3 uses accuracy.

| task | n | primary | accuracy | Brier / top-Brier ↓ | ECE / top-ECE ↓ | p50 ms | p95 ms | cost USD | Jev primary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| toxicchat | 600 | 0.977 | 0.930 | 0.054 | 0.021 | 309 | 864 | 0.0114 | 0.990 |
| beavertails | 500 | 0.874 | 0.780 | 0.168 | 0.133 | 237 | 743 | 0.0120 | 0.900 |
| halueval_qa | 600 | 0.329 | 0.377 | 0.420 | 0.399 | 242 | 645 | 0.0159 | 0.940 |
| halueval_dialogue | 600 | 0.807 | 0.733 | 0.182 | 0.055 | 266 | 697 | 0.0158 | 0.914 |
| boolq | 500 | 0.970 | 0.926 | 0.062 | 0.056 | 289 | 774 | 0.0130 | 0.977 |
| anli_r3 | 500 | 0.518 | 0.518 | 0.284 | 0.189 | 310 | 815 | 0.0122 | 0.662 |
| prompt_injection | 500 | 0.988 | 0.902 | 0.071 | 0.083 | 307 | 844 | 0.0108 | 0.994 |

- Macro mean primary metric: 0.781
- Total decisions: 3,800
- Total API cost: $0.0911
