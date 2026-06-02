"""Abstract storage interface (mimics a small slice of the GCS client surface)."""

from __future__ import annotations

from typing import Iterable, Protocol


class LogStorage(Protocol):
    """Minimal storage surface the rest of the codebase depends on.

    Designed so a future ``GCSBackend`` can implement the same protocol
    without any caller changes.
    """

    def write_text(self, key: str, body: str) -> None:
        """Write a UTF-8 text blob at ``key``. Parent directories are created."""
        ...

    def append_line(self, key: str, line: str) -> None:
        """Append a single line (no trailing newline expected) to ``key``."""
        ...

    def read_text(self, key: str) -> str:
        """Read a UTF-8 text blob at ``key``."""
        ...

    def iter_keys(self, prefix: str) -> Iterable[str]:
        """Yield keys under ``prefix`` (recursive)."""
        ...

    def exists(self, key: str) -> bool:
        """True if ``key`` exists."""
        ...
