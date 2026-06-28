from __future__ import annotations


class SemCacheError(Exception):
    """Raised when a SemCache operation fails and fail_open is disabled."""
