"""Backend package — re-exports for backward compatibility."""

from . import backend as _backend_module
from .backend import (
    LayeredBackend, logger,
    _retry_on_lock, _is_lock_error, _setup_logger, _LiveStderrHandler,
    _to_qdrant_id, _from_qdrant_id, _get_secret_getter, _setting,
    _pack_embedding, _unpack_embedding, _EMBED_NULL,
    _wrap_untrusted_text, _get_embedding_fn,
)


def __getattr__(name):
    """PEP 562 module __getattr__ — proxy _secret_getter live instead of
    snapshotting it at import time.

    backend.backend._get_secret_getter() reassigns that module's
    _secret_getter global on first resolution. A plain
    `from .backend import _secret_getter` (the old code here) copies the
    value present at package-import time — almost always None, since
    resolution hasn't happened yet — and that copy is never updated, so
    `from backend import _secret_getter` always saw the stale None.
    """
    if name == "_secret_getter":
        return _backend_module._secret_getter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [name for name in dir() if not name.startswith("_")] + ["_secret_getter"]
