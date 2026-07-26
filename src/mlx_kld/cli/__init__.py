"""mlx-kld command-line interface.

A single dispatcher, exposed as the ``mlx-kld`` console script and as
``python -m mlx_kld``. Subcommands:

  score        KLD-score a student checkpoint (MLX safetensors or GGUF)
               against a teacher
  compare      per-teacher comparison tables from the results root
  plot         SVG scatter of mean KLD vs bpw from the results root
  cache        inspect / prune the teacher-logit cache
  floor-sweep  measure the top-K reconstruction floor on a model
  self-test    run the numerical + schema unit suite (no model load)

The model-level subcommands keep their heavy imports (mlx, mlx_lm, datasets)
inside ``cmd`` so the dispatcher itself, and ``self-test``, stay light.
"""

from __future__ import annotations

import argparse
import os
import sys

from .. import __version__
from ..errors import MlxKldError, RecordSchemaError
from . import cache, compare, floor_sweep, plot, score, selftest

_SUBCOMMANDS = (score, compare, plot, cache, floor_sweep, selftest)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-kld",
        description="KL-divergence scoring of quantized student checkpoints "
        "(MLX safetensors or GGUF) against a full-precision teacher.",
    )
    parser.add_argument("--version", action="version", version=f"mlx-kld {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for mod in _SUBCOMMANDS:
        mod.add_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else list(argv)
    parser = _build_parser()
    args = parser.parse_args(raw)
    if not getattr(args, "func", None):
        # No subcommand: help is a diagnostic here, not the requested output,
        # so it goes to stderr with argparse's usage-error exit code. A
        # script that invokes `mlx-kld` bare should see a failure.
        parser.print_help(sys.stderr)
        return 2
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except RecordSchemaError as e:
        # A record that fails its own locked schema is a bug in this tool, not
        # something the user did. It subclasses MlxKldError so an embedding
        # caller can catch the whole family, so it has to be caught first here
        # to keep it from printing as an ordinary user error. Exit 70 is
        # sysexits.h EX_SOFTWARE: "an internal software error".
        if os.environ.get("MLX_KLD_DEBUG"):
            raise
        print(f"internal error: {e}\n"
              "This is a bug in mlx-kld. The record it built does not match "
              "its own schema. Please report it, with MLX_KLD_DEBUG=1 set for "
              "a traceback.", file=sys.stderr)
        return 70
    except MlxKldError as e:
        # A condition the library raises deliberately: the message is already
        # written for a user, so print it without the exception-type prefix.
        if os.environ.get("MLX_KLD_DEBUG"):
            raise
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        # Cold-user inputs (bad HF id, no network, unsupported arch) raise many
        # types; show a clean line, full traceback only under MLX_KLD_DEBUG.
        if os.environ.get("MLX_KLD_DEBUG"):
            raise
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
