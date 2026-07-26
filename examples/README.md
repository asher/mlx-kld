# Examples

Real output from real runs, committed so you can see what the tool produces
before installing it.

| Example | What it shows |
|---|---|
| [qwen3.6-27b/](qwen3.6-27b/) | 26 quantizations of one 27B model from 6 publishers, MLX and GGUF ranked in a single table |
| [floor-sweep-qwen3-0.6b.md](floor-sweep-qwen3-0.6b.md) | How the top-K measurement floor behaves as K changes, and why the default K is what it is |

Every number here came from the committed JSON records in
[qwen3.6-27b/records/](qwen3.6-27b/records/), which are ordinary
`schema_version=1` records. You can point the tool at them directly:

```bash
mlx-kld compare --out-dir examples/qwen3.6-27b/records --publisher
mlx-kld plot --out-dir examples/qwen3.6-27b/records --log-y --svg /tmp/chart.svg
```

Local filesystem paths in these records were rewritten to `models/...` before
committing. Nothing else was edited.
