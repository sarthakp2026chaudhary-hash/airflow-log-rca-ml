"""Local-filesystem backend that mimics a GCS bucket."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


class LocalFSBackend:
    """Treats a local directory as if it were a GCS bucket.

    Keys are forward-slash-separated paths relative to ``root``. They map
    1:1 onto on-disk paths, with parent directories created lazily.
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ----- key <-> path -----

    def _path(self, key: str) -> Path:
        # Reject path-escape attempts; GCS keys do not start with "/".
        clean = key.lstrip("/")
        if ".." in Path(clean).parts:
            raise ValueError(f"refusing key with parent traversal: {key!r}")
        return self.root / clean

    # ----- LogStorage protocol -----

    def write_text(self, key: str, body: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    def append_line(self, key: str, line: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read_text(self, key: str) -> str:
        return self._path(key).read_text(encoding="utf-8")

    def iter_keys(self, prefix: str) -> Iterable[str]:
        base = self._path(prefix)
        if not base.exists():
            return
        for p in base.rglob("*"):
            if p.is_file():
                yield p.relative_to(self.root).as_posix()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()
