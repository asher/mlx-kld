# mlx-kld

Measures how much a quantized model differs from the full-precision model it was
built from, using [MLX](https://github.com/ml-explore/mlx).

Quantization makes a model smaller by storing its weights at lower precision.
That changes the model's predictions. mlx-kld quantifies the change by running
both models over the same text and comparing, at every position, the probability
distribution each one assigns to the next token. The comparison uses
Kullback-Leibler divergence (KL divergence, abbreviated KLD below), a standard
measure of how far one probability distribution sits from another. Zero means
the two models predict identically. Larger means the quantized model has drifted
further from the original.

The full-precision model is called the **teacher** and the quantized one the
**student**, following the usual convention. mlx-kld reports the same metrics,
in the same units, as llama.cpp's `--kl-divergence`, so the numbers read on a
familiar scale. The absolute values are not interchangeable with llama.cpp's
own (see [Limitations](#limitations)).

Two things make it useful in practice. The teacher's predictions are cached to
disk, so scoring twenty students against one teacher pays the expensive teacher
computation once. And because teacher and student both run through MLX, a GGUF
file and an MLX checkpoint of the same model can be scored under one methodology
and ranked in one table.

**See it first:** [examples/qwen3.6-27b/](examples/qwen3.6-27b/) has 26
quantizations of one 27B model from six publishers, MLX and GGUF together, with
the chart and the underlying records.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Reading the numbers](#reading-the-numbers)
- [How scoring works](#how-scoring-works)
- [Choosing --top-k](#choosing---top-k)
- [Artifacts](#artifacts)
- [GGUF students](#gguf-students)
- [The teacher cache](#the-teacher-cache)
- [Running on a smaller machine](#running-on-a-smaller-machine)
- [Publishing results](#publishing-results)
- [Python API](#python-api)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)
- [Compatibility](#compatibility)

## Install

```bash
pip install mlx-kld

# also score K-quant checkpoints (loads them and computes bits-per-weight)
pip install 'mlx-kld[kquant]'

# also score GGUF files directly (see the dependency note below)
pip install 'mlx-kld[gguf]'
```

Requires Apple Silicon and Python 3.10 or newer (3.11+ for the `[gguf]` extra).
The teacher has to fit in unified memory alongside the student, so a 27B model
at bfloat16 (about 55 GB of weights) needs a 64 GB machine or larger. You also
need disk for the cache, which is sized in [The teacher cache](#the-teacher-cache).

Some terms used throughout: **HF** is Hugging Face, and an *HF id* is a
`org/name` model identifier like `Qwen/Qwen3-0.6B`. **bpw** is bits per weight,
the average storage cost of one parameter. **GGUF** is llama.cpp's model file
format. **K-quant** and **IQ** are families of quantization codecs used inside
GGUF files, and **MXFP4** is a 4-bit floating-point codec.

> **`[gguf]` dependency note.** The extra pulls in
> [gmlx](https://pypi.org/project/gmlx/), a full GGUF runtime, and gmlx's whole
> dependency set comes with it: several hundred MB, an exact pin on `mlx-vlm`
> (which brings opencv and scipy), and a serve stack. Scoring never touches the
> serve stack, and `mlx-vlm` is imported only for language-model class
> definitions on vision-language architectures, so no image code runs. The pins
> are strict, so install the `[gguf]` extra into a dedicated virtual environment
> rather than a shared one. Note also that gmlx is licensed under BUSL-1.1, a
> source-available license that is not open source. mlx-kld itself is Apache-2.0
> and a plain `pip install mlx-kld` depends on neither.

## Quick start

The teacher can be an HF id or a local path. The student must be local, so
download it first if it lives on the hub:

```bash
# On huggingface_hub older than 1.0 this command is `huggingface-cli download`.
hf download mlx-community/Qwen3-0.6B-4bit --local-dir ./Qwen3-0.6B-4bit

mlx-kld score Qwen/Qwen3-0.6B ./Qwen3-0.6B-4bit
```

That prints a Markdown report and writes a JSON record. More of the workflow:

```bash
# Score a GGUF file against the same teacher (needs the [gguf] extra).
# The two reports are directly comparable.
mlx-kld score Qwen/Qwen3-0.6B ./Qwen3-0.6B-Q4_K_M.gguf

# Also write a short block you can paste into a model card.
mlx-kld score Qwen/Qwen3-0.6B ./Qwen3-0.6B-4bit --card card.md

# Compare everything scored so far, one table per teacher, best first.
mlx-kld compare

# Chart one teacher's students: mean KLD against effective bpw.
mlx-kld plot --teacher Qwen3-0.6B --log-y --svg kld-vs-bpw.svg

# Use a different calibration corpus. sample-10BT is about 40 GB of parquet,
# so stream it rather than downloading the whole split.
mlx-kld score Qwen/Qwen3-0.6B ./Qwen3-0.6B-4bit \
    --dataset "HuggingFaceFW/fineweb-edu:sample-10BT" --stream-dataset

# Inspect and prune the teacher cache.
mlx-kld cache list
mlx-kld cache gc --max-gb 20

# Measure the top-K reconstruction floor to pick a --top-k.
mlx-kld floor-sweep Qwen/Qwen3-0.6B

# Numerical and schema unit suite (no model load).
mlx-kld self-test
```

Defaults: 512 sequences of 512 tokens from
`Salesforce/wikitext:wikitext-103-raw-v1`, top-k 32768, seed 123, scoring the
second half of each sequence. Chat-format corpora (those with a `messages`
column) are rendered through the scoring tokenizer's chat template before
tokenizing.

A run costs one forward pass over the full calibration set, which is 262,144
tokens at the default protocol. Scoring a student therefore takes about as long
as prefilling that many tokens through it: seconds to minutes for models under
1B, tens of minutes for a 27B. The first `score` against a new teacher costs
roughly twice that, because it also runs the teacher and writes the cache. Every
later student against the same teacher skips both.

## Reading the numbers

Mean KLD is in **nats**, the natural-logarithm unit of information. At each
scored position it measures how far the student's next-token distribution has
drifted from the teacher's, averaged over all scored tokens.

Exponentiating turns it into a likelihood ratio. `exp(mean KLD)` is the
geometric-mean factor by which the student under-weights what the teacher would
predict, and at the small values quantization produces, `e^x` is approximately
`1 + x`. So the mean KLD reads directly as a fractional per-token likelihood
penalty. A mean of 0.006 nats is about a 0.6 percent penalty, and 0.06 nats
about 6 percent. The same number is roughly how far the student's perplexity
sits above the teacher's, to the extent the teacher is well calibrated on this
corpus (see [Limitations](#limitations)).

**Rank students by mean KLD.** It measures the total cost of quantization.

The percentiles show how that cost is distributed across tokens. p50 is the
median token. p99 is the 99th percentile, meaning the threshold that only the
worst 1 percent of tokens exceed. A mean far above p50 means the mean is driven
by the tail.

Quantization damage is rarely uniform, and the worst 1 percent of tokens (rare
words, numbers, code syntax) is where generation visibly derails. The accounting
is direct. Every token in the worst 1 percent contributes at least p99 to the
sum, so the worst 1 percent carries at least `(p99 / mean)` percent of the total
cost. A ratio of 20 puts at least 20 percent of the cost in that 1 percent. A
ratio near 1 is a steady tax where every token costs about the same. A ratio
near the ceiling of 100 means the mean is an artifact of rare blowouts and the
median token is nearly perfect. Healthy quantizations of one model tend to share
a similar ratio, so a sibling whose ratio is an outlier is concentrating damage
somewhere even if its mean ranks well. At equal mean, prefer the lower p99,
especially for code, math, or function calling, where one bad token compounds.

The report also carries:

- **Delta-p**, the mean and root-mean-square (RMS) of `p_student - p_teacher` at
  the observed next token. The Markdown report renders it in percentage points,
  the unit llama.cpp prints, while the JSON record keeps raw probability.
- **SE**, the standard error on the mean, written as `mean +/- SE`. See
  [Limitations](#limitations) for how it is computed.
- **Top-1 agreement**, how often the two models' most likely tokens match. This
  is llama.cpp's "Same top p".
- **Top-5 agreement**, how often the teacher's top token appears anywhere in the
  student's top five.

## How scoring works

For each scored token, the teacher's top-K log-probabilities plus a uniform tail
covering the rest of the vocabulary reconstruct the teacher distribution `P`.
The student's full distribution is `Q`, and the per-token `KL(P||Q)` is computed
in fp32 in closed form. Working in closed form from the student's logits and a
per-position log-sum-exp avoids materializing the full `(batch, tokens, vocab)`
log-softmax.

The teacher pass also scores the teacher against its own reconstruction, which
gives the per-run measurement floor described below.

Defaults follow the conventional llama.cpp protocol: sequences of 512 tokens
(what llama.cpp calls `n_ctx`) with only the second half scored, so every scored
token has a substantial context behind it. `--long-context` switches to
full-sequence scoring at 2048 tokens.

A student whose output layer (`lm_head`, the final projection producing one
logit per vocabulary entry) is padded wider than the teacher's vocabulary is
sliced down, with a diagnostic reporting how much probability mass sat on the
padding. A genuinely narrower vocabulary is rejected with an explicit error.

## Choosing --top-k

Caching every logit for every position would tie the cache to the vocabulary
size. At the default protocol on a 151,936-token vocabulary that is about 80 GB
per teacher, which is roughly what llama.cpp's `--kl-divergence` base file
costs. Storing only the top K and modelling the rest as a uniform tail bounds
the entry by K instead.

That approximation has a cost. Even a student identical to its teacher scores a
small nonzero KLD, and that value is the **reconstruction floor**. Every report
carries the floor measured on that run's own cache, and both `score` and
`compare` flag a result as floor-limited when the mean KLD is within 2x of it.
Ranking floor-limited siblings against each other is still sound, because they
share the same floor. What to read carefully is the absolute magnitude, and any
ratio against a student that sits well above the floor.

**`--top-k` accepts any value up to the teacher's vocabulary size minus one.**
The default of 32768 is not a ceiling. It is, however, close to the best choice
on a typical large-vocabulary model, because **the floor is U-shaped in K**:

| K | floor on Qwen3-0.6B (nats) | cache entry |
|---:|---:|---:|
| 512 | 0.1258 | 0.8 GB |
| 2,048 | 0.0359 | 3.2 GB |
| 8,192 | 0.0069 | 12.9 GB |
| **32,768** | **0.0030** | **51.5 GB** |
| 65,536 | 0.0034 | 103.1 GB |
| 131,072 | 0.0056 | 206.2 GB |

Raising K past the minimum costs disk and accuracy at the same time. Two errors
move in opposite directions: a larger K shrinks the uniform-tail approximation,
but every cached log-probability is stored as bfloat16, so a larger K also
rounds more numbers. They swap dominance near the default.

The minimum's location depends on the vocabulary and on how peaked the model is,
so measure it for your own teacher rather than assuming this table transfers:

```bash
mlx-kld floor-sweep <teacher>
```

The floor's magnitude varies far more than its location. The 0.6B model above
floors at 0.0030 nats where a 27B floors at 0.0018 nats, both at K=32768. Full
data and method in
[examples/floor-sweep-qwen3-0.6b.md](examples/floor-sweep-qwen3-0.6b.md).

## Artifacts

The workflow is `score` once per student, then `compare`. Each `score` run
prints a Markdown report and writes a `schema_version=1` JSON record under the
results root, by default
`./kld-results/<teacher-slug>/<student-slug>.<digest8>.json`:

- `<teacher-slug>` is the teacher id lowercased with the org prefix kept, so two
  orgs' builds of one model name stay apart.
- `<student-slug>` is the student directory name, or the `.gguf` file's stem.
- `<digest8>` fingerprints the calibration spec, the tokenizer mode, the score
  window, and which teacher and student produced the run. Distinct runs never
  overwrite each other, and an identical rerun stays idempotent.

Each record also pins a content hash of the exact scored token stream, the
teacher's tokenizer hash, and, for hub ids, a best-effort revision. That makes a
silent upstream dataset or tokenizer change detectable later.

Model directories are never written to. Point `--out-dir`, or the
`MLX_KLD_RESULTS` environment variable, at a different results root if you want
one.

`compare` reads the same root and prints per-teacher tables sorted by mean KLD.
Runs whose calibration specs differ, or whose score windows differ, are never
blended into one ranking. They are grouped and flagged in a legend instead. The
token-stream hash is the comparability witness, so runs whose hashes match rank
in one table even when their tokenizer modes differ.

If the student directory holds a recipe sidecar named `quant-recipe.json`, its
scalar fields are folded into the record and report as free-form provenance.
Nothing else reads it, and its absence changes no measurement.

## GGUF students

With the `[gguf]` extra, `score` accepts a `.gguf` file as the student. The file
loads through gmlx's zero-conversion path, so the quantized bytes are executed
directly with no dequantize-and-requantize step, and the score reflects the GGUF
exactly as it ships. Teacher and student run the same MLX forward
implementation, so the KLD isolates the quantized weights rather than mixing in
engine differences. That is what makes GGUF and safetensors numbers from this
tool directly comparable. The dequantization kernels for the GGUF codecs are
gmlx's own, so "same engine" holds at the graph level rather than kernel for
kernel.

There are two tokenizer modes:

- **Default (synthesized):** the student tokenizer is rebuilt from the GGUF's
  embedded metadata, so the run tests the GGUF as shipped, tokenizer included.
  One caveat applies. A teacher cache entry is shared by every student scored
  against it, so a cache hit replays whichever token ids built that entry. When
  those differ from this GGUF's own tokenization, `score` says so and the record
  sets `tokenizer.stream_is_students` to `false`. Pass `--rebuild-cache` to
  score on this GGUF's own tokenization instead.
- **`--hf-source <id-or-dir>`:** the student borrows an HF tokenizer, normally
  the teacher's. Tokenization is then identical to the teacher's, so the test
  isolates the weights. This is also the escape hatch when gmlx cannot
  synthesize a tokenizer for an architecture.

In both modes an encoding-parity probe gates the run. If student tokenization
diverges from the teacher's the KLD would be meaningless, so scoring stops
unless `--allow-tokenizer-mismatch` forces it.

GGUF reports include `effective_bpw` computed from the per-tensor bytes in the
GGUF header, excluding file-level metadata overhead, so bits-per-weight is
comparable against safetensors students.

## The teacher cache

A cache entry is keyed by teacher, calibration spec, and top-k, and costs about
6 bytes per cached position per top-K slot. **The default configuration writes a
51.5 GB entry** on the first `score` against a new teacher, and `score` prints
the estimate before building one. An existing entry replays at its own batch
size, so `--batch-size` never silently forces a teacher rebuild.

Every position is cached, not just the scored window. Under the default
second-half protocol that is twice what any single run scores, roughly 25.8 GB
per entry that is read back without contributing to the score. In exchange,
`--score-window` and `--long-context` reshape the measurement without a teacher
rebuild. The score window is deliberately absent from the cache key, while the
record digest includes it, so results never blend across windows.

Concurrent runs against one cache root are safe. A per-entry file lock
serializes builds, so two simultaneous misses on one key build it once. `score`
holds that lock shared across the student pass, and every eviction path skips an
entry another process is using. `score` also re-reads the manifest once the
entry is pinned, so a rebuild that lands between building and replaying is
reported rather than silently mis-attributed.

To keep disk usage bounded:

- `score --cache-max-gb N` holds the cache root under `N` GB by evicting
  least-recently-used entries. Eviction runs before every teacher pass, to make
  room for the estimated new entry, and again after each run. `score` warns when
  the disk lacks the free space. The default of 120 keeps two default-config
  teachers resident, and `0` disables auto-eviction. The entry the current run
  uses is never evicted, and `score` warns when a single entry cannot fit the
  budget. That happens under `--long-context`, where an entry is about 206 GB.
- Lowering `--top-k` shrinks entries proportionally. See the table in
  [Choosing --top-k](#choosing---top-k).
- Lowering `--num-samples` shrinks them proportionally too.
- `mlx-kld cache list` shows each entry's size and last-used age.
- `mlx-kld cache gc --max-gb N --older-than DAYS` prunes on demand.

## Running on a smaller machine

The defaults assume you have the disk for a 51.5 GB cache entry. A smaller
protocol still produces usable rankings:

```bash
mlx-kld score <teacher> <student> --num-samples 128 --top-k 8192
```

That writes a 3.2 GB entry and scores 32,768 tokens instead of 131,072. Two
things get worse. The standard error widens, because a quarter as many tokens
are scored. And the reconstruction floor rises, on the 0.6B model above from
0.0030 to 0.0069 nats, so quantizations at 6 bits and above may come back
floor-limited.

Runs at a reduced protocol are internally consistent and rank correctly against
each other. They are not comparable against default-protocol runs, and `compare`
enforces that by grouping them separately.

The teacher itself is the harder constraint. It must fit in unified memory, and
no flag changes that.

## Publishing results

The JSON record is designed so that two people's runs can be merged into one
ranking. It pins the corpus, sample count, sequence length, seed, top-k, score
window, a content hash of the exact scored token stream, and the teacher's
tokenizer hash. `compare` groups on exactly those fields, so records that agree
rank together and records that do not are separated with a legend rather than
silently averaged.

To publish a result alongside a checkpoint:

```bash
mlx-kld score <teacher> <student> --card card.md
```

`--card` writes a short Markdown block with the headline metrics, the spec they
are comparable within, and the command that reproduces them. It deliberately
contains no local filesystem paths, so it is safe to paste into a model card.

To merge results from several people, put their JSON records in one directory
tree and run `compare` over it. `--pattern` accepts repeated globs if they live
in different places:

```bash
mlx-kld compare --pattern 'mine/**/*.json' --pattern 'theirs/**/*.json'
```

For results to land in one table rather than separate groups, everyone has to
run the same teacher on the same protocol. Since absolute values are not
comparable against llama.cpp's published numbers, agreeing on a shared teacher
and protocol is what makes cross-publisher comparison meaningful.

## Python API

Tools that generate checkpoints can drive the scorer in-process rather than
shelling out:

```python
from mlx_kld import (
    ensure_teacher_topk_cache, score_loaded_student, entry_lock, MlxKldError,
)

cache_dir, manifest, tokenizer = ensure_teacher_topk_cache(
    teacher_path="Qwen/Qwen3-0.6B",
    dataset_name="Salesforce/wikitext:wikitext-103-raw-v1",
    num_samples=512, max_seq_len=512, seed=123, top_k=8192,
)

# Hold the entry against eviction by another process for the whole replay, and
# re-read manifest.json once you hold the lock.
with entry_lock(cache_dir.parent, cache_dir.name, exclusive=False):
    metrics = score_loaded_student(model, cache_dir, manifest)

print(metrics["kld"]["mean"])
```

Every deliberate failure raises a subclass of `MlxKldError`
(`CacheMismatchError`, `TokenizerMismatchError`, `CalibrationCorpusError`,
`RecordSchemaError`), so an embedding caller can catch the family without
catching bugs. The library never calls `sys.exit`.

One thing to know: the cache key does not cover the tokenizer, so one entry is
shared by every student at that teacher and spec. Whoever builds it first fixes
the token stream everyone else replays. The manifest's `corpus_tokens_hash` is
the witness. Compare it against a stream you tokenized yourself before trusting
that the cache matches your intent, and pass `rebuild=True` when it does not.

## Troubleshooting

**"TOKENIZER ENCODING DIVERGES"** means the two tokenizers turn the same text
into different token ids, so the teacher would be scored on a tokenization it
never saw. For a GGUF student, pass `--hf-source <teacher>` to borrow the
teacher's tokenizer and test the weights alone. Otherwise the checkpoints
genuinely disagree and the comparison is not meaningful.
`--allow-tokenizer-mismatch` forces past it, and the record records that it was
forced.

**"WARNING - implausible result"** means top-1 agreement fell below 50 percent
or mean KLD rose above 1.0 nats. Real quantizations do not land there. Suspect a
broken load, a wrong normalization convention, a vocabulary mismatch, or corrupt
weights, and treat the score as invalid until you find the cause.

**"scored stream is NOT this student's tokenization"** means a cache hit replayed
token ids that an earlier student built the entry from. The weights under test
are this student's, the tokenization is not. Rescore with `--rebuild-cache`.

**"Floor-limited"** means the result is within 2x of the measurement floor. The
ranking against siblings still holds. See
[Choosing --top-k](#choosing---top-k).

**"tokens dropped (non-finite KLD)"** means the student produced infinite or NaN
logits at some positions. Those positions leave the mean rather than poison it,
which makes the number look better than reality, so any nonzero count deserves
investigation.

Set `MLX_KLD_DEBUG=1` for a full traceback instead of a one-line error.

## Limitations

Every one of these is a property of the protocol rather than a bug, and each is
recorded in the JSON record so a reader can see it rather than infer it.

- **The calibration set is drawn from the head of the corpus.** Ingest stops as
  soon as it holds twice the tokens it needs, about 525k at the default
  protocol, so `--seed` permutes within that window rather than sampling the
  corpus as a whole. On wikitext-103 (about 103M tokens) that window is the
  first 0.5 percent, a few dozen articles. Runs stay reproducible and comparable
  to each other, which is what the protocol needs, but a different `--seed` is
  not a different sample of the corpus.
- **Absolute KLD carries the top-K reconstruction floor.** The floor is measured
  on every run and reported alongside the mean, and a mean within 2x of it is
  flagged. Siblings that share a cache share a floor, so their ranking is
  unaffected.
- **The standard error is clustered by sequence.** The 256 scored tokens inside
  one calibration sequence share a document and a context, so they are not
  independent draws. A naive `std/sqrt(n)` over the 131k tokens of a default run
  would understate the error. The reported SE uses a cluster-robust CR1
  estimator over sequences, a standard correction for exactly this situation,
  which comes out several times wider. It is still slightly optimistic, because
  sequences are fixed-length slices of one concatenated stream and several can
  come from the same document. The record's `kld.se_method` names the form that
  produced the number.
- **The perplexity reading is an approximation.** The perplexity gap averages
  `log p_teacher - log p_student` over the *data* distribution, while KLD
  averages the same quantity over the *teacher's*. The two coincide only when
  the teacher is the data-generating distribution. Read the perplexity
  interpretation as holding to the extent the teacher is well calibrated on this
  corpus.
- **Not interchangeable with llama.cpp's published numbers.** llama.cpp measures
  against an f16 GGUF teacher under a different engine and windowing. The metric
  family and the units match, so the scales are familiar, but the absolutes are
  two different measurements.
- **Delta-p covers one fewer position per sequence than KLD**, 255 against 256
  under the default window, because the last position in a sequence has no
  observed next token to compare against.
- **Positions the student cannot be scored on are dropped, not counted.** A
  student that emits non-finite logits somewhere yields a non-finite KLD there.
  Those positions leave the mean rather than poison it, which *improves* the
  number. The count is recorded as `tokens_dropped_nonfinite` and flagged in the
  report, so a nonzero value is visible rather than inferred.
- **Text-only.** No vision or audio models in v0.1.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version: `ruff check` and
`pytest` must be clean, `ruff format` is deliberately not the style, and a
change that can move a reported number should say so in the pull request.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
