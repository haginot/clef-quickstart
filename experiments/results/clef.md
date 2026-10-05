# cloudflare/clef benchmark result

One request per item. Binary tasks use AUROC as the primary metric; ANLI R3 uses accuracy.

| task | n | primary | accuracy | Brier / top-Brier ↓ | ECE / top-ECE ↓ | p50 ms | p95 ms | cost USD | Jev primary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| toxicchat | 600 | 0.986 | 0.940 | 0.047 | 0.033 | 666 | 1362 | 0.0304 | 0.990 |
| beavertails | 500 | 0.905 | 0.832 | 0.128 | 0.086 | 592 | 1114 | 0.0320 | 0.900 |
| halueval_qa | 600 | 0.856 | 0.705 | 0.209 | 0.201 | 545 | 1093 | 0.0423 | 0.940 |
| halueval_dialogue | 600 | 0.896 | 0.717 | 0.193 | 0.183 | 568 | 1114 | 0.0423 | 0.914 |
| boolq | 500 | 0.975 | 0.932 | 0.055 | 0.042 | 560 | 1058 | 0.0347 | 0.977 |
| anli_r3 | 500 | 0.534 | 0.534 | 0.286 | 0.206 | 597 | 1111 | 0.0326 | 0.662 |
| prompt_injection | 500 | 0.988 | 0.934 | 0.050 | 0.043 | 605 | 1066 | 0.0287 | 0.994 |

- Macro mean primary metric: 0.877
- Total decisions: 3,800
- Total API cost: $0.2429
