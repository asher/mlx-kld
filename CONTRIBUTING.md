# Contributing

Thanks for looking. This is a measurement tool, so the bar for changes that
move a reported number is deliberately higher than the bar for changes that do
not. That distinction shapes most of what follows.

## Getting set up

```bash
git clone https://github.com/asher/mlx-kld
cd mlx-kld
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

Apple Silicon and Python 3.10+ are required (3.11+ for the `[gguf]` extra).
`pytest` works from a fresh clone without the editable install as well, since
the repo's `conftest.py` puts `src/` on the path for both in-process imports
and the subprocess tests.

## Before you open a pull request

```bash
ruff check src tests conftest.py   # must be clean
pytest -q                          # ~280 tests, no model load
mlx-kld self-test                  # the numerical + schema suite the CLI ships
```

CI runs the same checks. `ruff check` runs on Linux, and the test suite plus
`self-test` run on macOS against Python 3.10, 3.11, 3.12, and 3.13.
The release workflow builds and publishes on a `vX.Y.Z` tag, which must match
`__version__` in `src/mlx_kld/__init__.py`.

## Style

- `ruff check` is the style. `ruff format` is deliberately **not** adopted.
  Several modules use hand-aligned inline comments to annotate tensor shapes
  (see `kld_math.py`), and the formatter destroys that alignment. Please do not
  run it, and please do not add it in a pull request that also changes
  behaviour.
- Lint rules are `["E", "W", "F", "I", "UP", "B"]` at a 100-column limit. The
  set is in `pyproject.toml`.
- Comments should say *why*, not *what*. Several comments in this codebase
  exist to stop a future reader from "fixing" something that is correct on
  purpose. `kld_math.kld_from_topk`'s note on error cancellation is the
  clearest example. If you find one of those, please leave it and its
  reasoning intact.

### Prose

The reports and tables this tool prints are what most users see, and the docs
are written to match them.

- **ASCII only**, in code and documentation alike. Write `+/-` rather than a
  sign glyph, `x` rather than a multiplication sign, and no smart quotes. The
  few lines that genuinely need non-ASCII, such as the tokenizer encoding
  probes, carry a trailing `# ascii-exempt` marker.
- **Do not punctuate a sentence with a double hyphen.** Write two sentences, or
  use a comma. Flag names like `--top-k` are obviously fine.
- **Go easy on semicolons and colons.** One per paragraph is plenty. A colon
  introducing a list is fine, but a colon splicing two independent clauses
  usually reads better as a full stop.
- Define an acronym at first use in user-facing text. A reader who has not been
  in this codebase does not know what bpw, MTP, or CR1 mean.
- Keep the documentation factual. Numbers in the docs should be reproducible
  from a command in the docs.

## Changes that move a number

Anything touching `kld_math.py`, `scoring.py`, `cache.py`'s teacher pass, or
`tokenizer.py`'s calibration ingest can change every score the tool has ever
produced. For those:

1. Say in the pull request whether the change is intended to move results. If
   it is not, show that it does not. A hash comparison, or a before/after on a
   real run, is worth more than an argument.
2. Watch the artifact contracts. `manifest["corpus_tokens_hash"]`,
   `tokenizer_hash`, and `run_digest` are all witnesses that let a record be
   compared against another record. Changing what any of them hashes
   invalidates existing caches and splits existing results into
   non-comparable groups.
3. Bump `CACHE_FORMAT_VERSION` in `_constants.py` if a cache entry written by
   the new code cannot be read correctly by the old, or vice versa.
4. Leave `SCHEMA_VERSION = 1` alone. The locked record keys are a contract with
   downstream consumers. Add non-locked keys alongside them instead.

## Tests

New behaviour needs a test, and the suite is built so that this costs almost
nothing. Nothing in it loads a model or touches the network. `teacher_pass`
runs against a fake model (`tests/test_teacher_pass.py`), calibration ingest
against a fake dataset and tokenizer (`tests/test_load_calibration.py`), and
the CLI through subprocesses that exit before any model load.

Prefer a test that states the property in its name and asserts the consequence
a user would notice, over one that pins an implementation detail.

## Examples

`examples/` holds real output from real runs, committed so a reader can see
what the tool produces before installing it. If you change the report, table, or
chart format, regenerate the affected files from the committed records:

```bash
mlx-kld compare --out-dir examples/qwen3.6-27b/records --publisher --scored-bpw \
    --md examples/qwen3.6-27b/comparison.md
mlx-kld plot --out-dir examples/qwen3.6-27b/records --log-y \
    --svg examples/qwen3.6-27b/kld-vs-bpw-log.svg
```

Records committed under `examples/` must carry no local filesystem paths.

## Reporting a wrong number

If a score looks wrong, the JSON record has what is needed to diagnose it.
Please include it, or at least its `calibration`, `tokenizer`, `cache`, and
`plausibility` blocks, plus the `mlx-kld --version` output and the versions of
`mlx`, `mlx-lm`, and (if relevant) `gmlx` or `mlx-kquant`.

## License

By contributing you agree that your contributions are licensed under the
Apache License 2.0, the same as the rest of the project.
