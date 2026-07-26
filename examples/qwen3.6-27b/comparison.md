# KLD comparison: `mlx-community__Qwen3.6-27B-bf16`

- teacher: `models/mlx/mlx-community__Qwen3.6-27B-bf16`
- precision: `bfloat16`
- runs: **26**
- generated: 2026-07-26T08:31:27+00:00

All runs share one spec: corpus=`Salesforce/wikitext:wikitext-103-raw-v1`, num_samples=512, max_seq_len=512, seed=123, top_k=32768, window=256:512, tokenizer=hf-source, mlx-student, tokens=3862684d4e318b3b.

| student | publisher | format | size | bpw | scored bpw | mean KLD +/- se | p99 | top-1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `mlx-community__Qwen3.6-27B-bf16` | mlx-community | mlx | 54.71 GB | 16.000 | 16.000 | 0.0018 +/- 7.6e-06 (f) | 0.0129 | 100.00% |
| `Qwen3.6-27B-UD-Q6_K_XL.gguf` | unsloth | gguf | 26.02 GB | 7.615 | 7.622 | 0.0048 +/- 0.0003 | 0.0258 | 98.01% |
| `Qwen_Qwen3.6-27B-Q6_K_L.gguf` | bartowski | gguf | 24.29 GB | 7.110 | 7.088 | 0.0052 +/- 0.0002 | 0.0268 | 97.94% |
| `Qwen_Qwen3.6-27B-Q6_K.gguf` | bartowski | gguf | 23.68 GB | 6.929 | 6.905 | 0.0059 +/- 0.0003 | 0.0278 | 97.64% |
| `Qwen3.6-27B-Q6_K.gguf` | unsloth | gguf | 22.88 GB | 6.698 | 6.696 | 0.0065 +/- 0.0004 | 0.0312 | 97.52% |
| `Qwen_Qwen3.6-27B-Q5_K_L.gguf` | bartowski | gguf | 21.75 GB | 6.366 | 6.332 | 0.0082 +/- 0.0003 | 0.0557 | 97.11% |
| `deepsweet__Qwen3.6-27B-MLX-oQ6` | deepsweet | mlx-affine | 22.49 GB | 6.690 | 6.690 | 0.0087 +/- 0.0004 | 0.0523 | 97.16% |
| `Qwen_Qwen3.6-27B-Q5_K_M.gguf` | bartowski | gguf | 20.97 GB | 6.136 | 6.098 | 0.0087 +/- 0.0003 | 0.0573 | 96.86% |
| `mlx-community__Qwen3.6-27B-6bit` | mlx-community | mlx-affine | 22.78 GB | 6.661 | 6.501 | 0.0091 +/- 0.0004 | 0.0534 | 96.89% |
| `Qwen3.6-27B-UD-Q5_K_XL.gguf` | unsloth | gguf | 20.35 GB | 5.956 | 5.957 | 0.0094 +/- 0.0004 | 0.0597 | 96.86% |
| `Qwen3.6-27B-Q5_K_M.gguf` | unsloth | gguf | 19.83 GB | 5.805 | 5.800 | 0.0095 +/- 0.0005 | 0.0604 | 96.76% |
| `Qwen3.6-27B-Q5_K_S.gguf` | unsloth | gguf | 19.27 GB | 5.639 | 5.636 | 0.0105 +/- 0.0006 | 0.0690 | 96.66% |
| `Qwen3.6-27B-UD-Q4_K_XL.gguf` | unsloth | gguf | 17.91 GB | 5.241 | 5.244 | 0.0182 +/- 0.0004 | 0.1662 | 95.27% |
| `deepsweet__Qwen3.6-27B-MLX-VL-oQ5` | deepsweet | mlx-affine | 19.92 GB | 5.825 | n/a | 0.0196 +/- 0.0005 | 0.1523 | 94.69% |
| `Qwen3.6-27B-Q4_K_M.gguf` | unsloth | gguf | 17.11 GB | 5.006 | 4.999 | 0.0204 +/- 0.0005 | 0.1851 | 94.86% |
| `Qwen_Qwen3.6-27B-Q4_K_S.gguf` | bartowski | gguf | 16.93 GB | 4.953 | 4.897 | 0.0217 +/- 0.0008 | 0.2114 | 94.51% |
| `mlx-community__Qwen3.6-27B-5bit` | mlx-community | mlx-affine | 19.42 GB | 5.678 | n/a | 0.0218 +/- 0.0006 | 0.1663 | 94.57% |
| `Qwen3.6-27B-Q4_K_S.gguf` | unsloth | gguf | 15.86 GB | 4.713 | 4.713 | 0.0227 +/- 0.0008 | 0.2208 | 94.42% |
| `Qwen3.6-27B-IQ4_XS.gguf` | unsloth | gguf | 15.44 GB | 4.589 | 4.589 | 0.0242 +/- 0.0009 | 0.2411 | 94.36% |
| `Jundot__Qwen3.6-27B-oQ4e-mtp` | Jundot | mlx-affine | 17.01 GB | 4.898 | n/a | 0.0397 +/- 0.0006 | 0.3553 | 91.69% |
| `Qwen3.6-27B-UD-Q3_K_XL.gguf` | unsloth | gguf | 14.47 GB | 4.302 | 4.302 | 0.0474 +/- 0.0017 | 0.5008 | 91.94% |
| `mlx-community__Qwen3.6-27B-4bit` | mlx-community | mlx-affine | 16.05 GB | 4.695 | n/a | 0.0573 +/- 0.0008 | 0.5292 | 89.96% |
| `Qwen_Qwen3.6-27B-IQ3_M.gguf` | bartowski | gguf | 14.12 GB | 4.130 | 4.061 | 0.0707 +/- 0.0021 | 0.7754 | 89.96% |
| `Qwen3.6-27B-Q3_K_S.gguf` | unsloth | gguf | 12.36 GB | 3.673 | 3.673 | 0.0927 +/- 0.0025 | 1.0268 | 88.07% |
| `bearzi__Qwen3.6-27B-oQ3` | bearzi | mlx-affine | 13.16 GB | 3.849 | 3.641 | 0.2092 +/- 0.0039 | 1.9662 | 79.94% |
| `NexVeridian__Qwen3.6-27B-3bit` | NexVeridian | mlx-affine | 11.77 GB | 3.501 | 3.501 | 0.2154 +/- 0.0040 | 2.0194 | 79.66% |

(f) floor-limited: mean KLD within 2x of the top-K reconstruction floor, so much of the absolute value is measurement floor. These rows still rank correctly against their siblings, which carry the same floor.

## Sources

- `examples/qwen3.6-27b/records/mlx-community__Qwen3.6-27B-bf16.fb0f8389.json`
- `examples/qwen3.6-27b/records/Qwen3.6-27B-UD-Q6_K_XL.a46df3c6.json`
- `examples/qwen3.6-27b/records/Qwen_Qwen3.6-27B-Q6_K_L.de66bac8.json`
- `examples/qwen3.6-27b/records/Qwen_Qwen3.6-27B-Q6_K.de66bac8.json`
- `examples/qwen3.6-27b/records/Qwen3.6-27B-Q6_K.971842ee.json`
- `examples/qwen3.6-27b/records/Qwen_Qwen3.6-27B-Q5_K_L.de66bac8.json`
- `examples/qwen3.6-27b/records/deepsweet__Qwen3.6-27B-MLX-oQ6.fb0f8389.json`
- `examples/qwen3.6-27b/records/Qwen_Qwen3.6-27B-Q5_K_M.de66bac8.json`
- `examples/qwen3.6-27b/records/mlx-community__Qwen3.6-27B-6bit.fb0f8389.json`
- `examples/qwen3.6-27b/records/Qwen3.6-27B-UD-Q5_K_XL.de66bac8.json`
- `examples/qwen3.6-27b/records/Qwen3.6-27B-Q5_K_M.7ffaf09e.json`
- `examples/qwen3.6-27b/records/Qwen3.6-27B-Q5_K_S.6bf1beb6.json`
- `examples/qwen3.6-27b/records/Qwen3.6-27B-UD-Q4_K_XL.de66bac8.json`
- `examples/qwen3.6-27b/records/deepsweet__Qwen3.6-27B-MLX-VL-oQ5.fb0f8389.json`
- `examples/qwen3.6-27b/records/Qwen3.6-27B-Q4_K_M.de66bac8.json`
- `examples/qwen3.6-27b/records/Qwen_Qwen3.6-27B-Q4_K_S.c9d3b089.json`
- `examples/qwen3.6-27b/records/mlx-community__Qwen3.6-27B-5bit.fb0f8389.json`
- `examples/qwen3.6-27b/records/Qwen3.6-27B-Q4_K_S.7e889724.json`
- `examples/qwen3.6-27b/records/Qwen3.6-27B-IQ4_XS.1194f457.json`
- `examples/qwen3.6-27b/records/Jundot__Qwen3.6-27B-oQ4e-mtp.fb0f8389.json`
- `examples/qwen3.6-27b/records/Qwen3.6-27B-UD-Q3_K_XL.94b2d936.json`
- `examples/qwen3.6-27b/records/mlx-community__Qwen3.6-27B-4bit.fb0f8389.json`
- `examples/qwen3.6-27b/records/Qwen_Qwen3.6-27B-IQ3_M.098dae12.json`
- `examples/qwen3.6-27b/records/Qwen3.6-27B-Q3_K_S.703f25be.json`
- `examples/qwen3.6-27b/records/bearzi__Qwen3.6-27B-oQ3.4252fa81.json`
- `examples/qwen3.6-27b/records/NexVeridian__Qwen3.6-27B-3bit.46bd0fb2.json`
