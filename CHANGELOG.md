# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

First release.

### Added

**Scoring**

- `mlx-kld score`: KL-divergence scoring of a quantized student against a
  full-precision teacher, with disk-cached top-K-truncated teacher logits.
  Headline metrics follow llama.cpp's `--kl-divergence` in family and units:
  mean KLD +/- SE, percentiles, the Delta-p family (mean +/- SE, RMS), and
  top-1 ("Same top p") / top-5 agreement. The Markdown report renders Delta-p
  in percentage points, as llama.cpp does, while the JSON record keeps raw
  probability. Absolutes are not interchangeable with llama.cpp's own, which
  come from a different engine and teacher.
- Standard errors are cluster-robust (CR1), clustered by calibration sequence.
  Tokens inside one sequence share a document and a context, so the naive
  `std/sqrt(n)` over the 131k scored tokens of a default run understates the
  error by the design effect. `kld.se_method` in the record names the form that
  produced the number.
- Per-run reconstruction floor: the teacher pass scores the teacher against its
  own cached top-K reconstruction, and the report carries the floor next to the
  mean KLD (`kld.floor_mean`), with an explicit floor-limited caution when the
  mean is within 2x of it. `compare` marks floor-limited rows.
- `mlx-kld floor-sweep`: measure the top-K reconstruction floor to pick a
  defensible `--top-k`. It sweeps the scored window (second half by default,
  `--score-window` to override), marks the K with the lowest floor, warns when
  the floor rises again above that K, and states the per-position cache cost
  there. The default K values span both sides of the default `--top-k`, because
  the floor is U-shaped in K rather than monotone: below the minimum the
  uniform-tail approximation dominates, and above it bfloat16 rounding on the
  stored slots does.
- Padded student lm_heads (a wider logits row than the teacher's vocab) are
  sliced to the teacher vocab with a padded-mass diagnostic instead of a hard
  error, while a genuinely narrower vocab is rejected with an explicit error.
- Result-plausibility guard: a run whose top-1 agreement and mean KLD together
  indicate a broken load, rather than a lossy quantization, is flagged in the
  report and in the record.

**The record**

- A locked `schema_version=1` JSON record per run, written under a results root
  (`./kld-results/<teacher>/<student>.<digest8>.json`, overridable via
  `--out-dir` / `MLX_KLD_RESULTS`), never into model directories. `digest8`
  fingerprints calibration, tokenizer mode, score window, and run identity
  (which teacher, which student path), so distinct runs never overwrite each
  other while an identical rerun stays idempotent. Records are written
  atomically, so an interrupted run cannot leave a truncated one behind. The
  record carries a per-format `quantization` descriptor (affine bits and group
  size, K-quant and GGUF codec histograms), `effective_bpw`, `n_params`, and
  full tokenizer-parity provenance.
- Reproducibility pins in the record: `calibration.corpus_tokens_hash` (a
  content hash of the exact scored token stream, which catches silent upstream
  dataset drift), `teacher.tokenizer_hash`, and a best-effort teacher
  `revision` for hub ids. A null revision is preserved as null rather than
  looked up at score time, since resolving it later would pin whatever the hub
  currently serves onto logits captured earlier.
- `teacher.precision` is measured. The teacher pass reads the dominant weight
  dtype off the loaded model and stores it in the manifest, so the field is a
  fact about the teacher rather than a label. `--teacher-precision` overrides it
  for a manifest that does not carry one.
- `calibration.top_k` and `cache.top_k` carry the K the teacher pass actually
  cached, read back from the manifest, not the K requested on the command line.
  The two differ whenever `--top-k` is clamped against the teacher vocab.
- `tokenizer.stream_is_students`: a teacher cache entry is shared by every
  student scored against it, so a cache hit replays whichever token ids built
  the entry. When those differ from the student's own tokenization, `score`
  warns and this field is `false`. `--rebuild-cache` scores on the student's
  own stream.
- `tokens_dropped_nonfinite` (non-locked): positions dropped because the student
  produced a non-finite KLD there. Dropping the worst positions improves the
  mean, so the count is recorded rather than only warned about, and the Markdown
  report flags it when nonzero.
- A free-form provenance sidecar reader. If the student directory holds
  `quant-recipe.json` or `recipe.json`, any scalar field in it is carried into
  the record. Nested structures are dropped, and key count and string length are
  bounded, because the file is another tool's output.
- `score --card PATH`: a short Markdown block sized for a model card, carrying
  the headline metrics, the calibration spec they are comparable within, and the
  command that reproduces them. It contains no local filesystem paths, so it is
  safe to paste into a published model card.

**Comparing and plotting**

- `mlx-kld compare`: per-teacher comparison tables from the results root, sorted
  by mean KLD. Runs with different specs (corpus, samples, top-k, score window,
  or scored-token content hash) are never blended into one ranking, and group
  under a spec legend instead. Runs whose content hash matches rank in one table
  even across tokenizer modes, since an identical scored stream is comparable
  regardless of how it was produced. Records without the hash split on tokenizer
  mode as before.
- `compare --publisher` and `compare --scored-bpw` columns, backed by the
  record's `scored_bpw`, `scored_n_params`, and `scored_bytes` fields. Scored
  bits per weight covers only the weights the scoring pass loads, excluding any
  multi-token-prediction stack and any vision or audio tower, so multi-stack
  checkpoints compare on the weights actually measured.
- `mlx-kld plot`: dependency-free SVG scatter of one teacher's students,
  X = effective bits per weight (`--x size` for file size), Y = mean KLD
  (`--log-y`), marker shape and color per student format, with per-point student
  labels and the reconstruction floor drawn in. Unquantized baselines are
  excluded, and mixed run specs are refused, mirroring `compare`'s
  comparability rule.

**The teacher cache**

- `mlx-kld cache {list,gc,clear}` plus `score --cache-max-gb`: a self-managing
  teacher cache that tracks per-entry size and last-used time, and evicts
  least-recently-used entries to stay under a budget. Eviction runs before every
  teacher pass, making room for the estimated new entry (with a free-disk-space
  warning), and again after each run. `cache clear` without a `--teacher` filter
  requires `--yes`. An existing cache entry replays at its own batch size, so
  `--batch-size` never silently forces a teacher rebuild.
- Concurrent runs against one cache root are safe. An advisory per-entry file
  lock serializes builds, so two simultaneous misses on one key build it once.
  `score` holds the lock shared across the student pass and re-reads the
  manifest once the entry is pinned, and every eviction path (auto-budget,
  `cache gc`, `cache clear`) skips entries another live process holds.
- A local teacher path is canonicalized before it enters the cache key, so
  `./teacher`, `teacher/`, `~/me/teacher`, and a symlinked ancestor all name one
  entry instead of up to four. Hub ids (`org/name`) keep their own keys. One
  consequence is worth knowing: mlx-lm resolves a bare `org/name` as a *local
  directory* when one exists relative to the current working directory, and the
  cache key follows that resolution so the key always matches the model actually
  loaded. Running from a directory that happens to contain `./Qwen/Qwen3-0.6B`
  therefore keys differently than running from anywhere else. Use an absolute
  path if you keep mirrors in that layout.

**Formats and data sources**

- Optional `[kquant]` extra to score K-quant safetensors checkpoints (loading
  and bits per weight via `mlx-kquant`).
- Optional `[gguf]` extra to score GGUF students (K-quant, IQ, and MXFP4
  codecs) directly through the `gmlx` zero-conversion loader: `mlx-kld score
  <teacher> model.gguf`. The student tokenizer is synthesized from GGUF metadata
  by default, and `--hf-source` borrows an HF tokenizer instead for a
  weights-only test. Requires Python 3.11+ (see the README's dependency note).
- `score --stream-dataset` and `floor-sweep --stream-dataset`: pull calibration
  rows over the network instead of downloading the whole split first, which is
  what makes web-scale corpora usable (fineweb-edu's sample-10BT is about
  40 GB). Same rows in the same order, so the token stream and its hash are
  unchanged.
- Vendored the mlx-lm PR #990 qwen3_5 sanitize fix (norm shift gated on
  unsanitized conv1d state only, not on leftover mtp weights), so
  already-converted Qwen3.5 and Qwen3.6 MLX checkpoints carrying mtp tensors no
  longer load with double-shifted norms. To be dropped once a released mlx-lm
  includes the fix.

**Interfaces and operations**

- A public exception hierarchy (`MlxKldError` and subclasses) for use as an
  embedded Python API, in place of `sys.exit()` calls inside the library. The
  CLI translates them to exit codes.
- Bare `mlx-kld` with no subcommand prints help to stderr and exits 2, so a
  scripted caller sees "no command given" as a failure.
- A record that fails its own locked schema exits 70 (`EX_SOFTWARE`) and prints
  `internal error:`. That failure means this tool built a bad record. It is not
  something the user typed, and it should not be indistinguishable from a bad
  flag.
- `mlx-kld self-test`: the numerical and schema unit suite, with no model load.
- `examples/`: real output from real runs, committed so the tool's output can be
  read before installing it. `examples/qwen3.6-27b/` holds 26 quantizations of
  one 27B model from six publishers with the underlying JSON records, and
  `examples/floor-sweep-qwen3-0.6b.md` documents how the measurement floor
  behaves as top-K changes.
- GitHub Actions workflows. `test.yml` runs the suite and `self-test` on macOS
  against every Python version the package advertises (3.10 through 3.13), with
  an import canary and a wheel-completeness check, plus a pinned-ruff lint job
  on Linux. `release.yml` builds and publishes to PyPI through trusted
  publishing on a `vX.Y.Z` tag, gated on the tag matching `__version__`.

### Known limitations

- Text-only in v0.1. Multimodal (VLM) scoring is reserved for a `[vlm]` extra in
  a later release.
- `student.size_bytes` means on-disk weight-file bytes in every format
  (`*.safetensors` for MLX, the `.gguf` shards for GGUF), while `effective_bpw`
  is computed from tensor wire bytes so container overhead cannot inflate a bit
  width. See the README's Limitations section for the protocol properties that
  shape every number: head-window calibration, the top-K reconstruction floor,
  and the perplexity approximation.
