"""Exceptions raised deliberately by the library layer.

``mlx_kld`` exposes a Python API (:func:`~mlx_kld.ensure_teacher_topk_cache`,
:func:`~mlx_kld.score_loaded_student`) that external quantizer tooling drives
in-process. A library cannot signal those conditions with ``sys.exit``: the
caller sees ``SystemExit``, which derives from ``BaseException`` and so slips
past any ordinary ``except Exception`` handler, taking the host process down
with it.

Every deliberate, user-facing failure therefore raises an
:class:`MlxKldError`. The CLI catches it in one place and renders it as a clean
one-line message with exit code 1, so the terminal experience is unchanged.
"""

from __future__ import annotations


class MlxKldError(Exception):
    """Base for every error mlx-kld raises deliberately.

    Catch this to handle any expected failure; catch a subclass to distinguish.
    Bugs still surface as ordinary exceptions.
    """


class CacheMismatchError(MlxKldError):
    """A cache entry does not match what this run needs.

    Raised when an entry was built against a different teacher tokenizer, so
    replaying it would score against logits the teacher never produced.
    """


class TokenizerMismatchError(MlxKldError):
    """Teacher and student do not agree on the tokenization KLD requires.

    Covers a genuinely narrower student vocabulary and a missing chat template.
    A metadata-only difference is reported, not raised. See
    :func:`~mlx_kld.tokenizer.encoding_parity`.
    """


class RecordSchemaError(MlxKldError):
    """A JSON record does not satisfy the locked ``schema_version=1`` contract.

    Raised by the pre-write validator (a scorer bug surfaces loudly instead of
    writing a malformed record) and caught by ``compare``/``plot`` readers,
    which warn-skip foreign or truncated files instead of crashing.
    """


class CalibrationCorpusError(MlxKldError):
    """The calibration corpus cannot supply the requested protocol.

    Raised for a corpus with no usable text column, a chat corpus whose rows
    fail to render, or one too small to yield ``num_samples`` chunks.
    """
