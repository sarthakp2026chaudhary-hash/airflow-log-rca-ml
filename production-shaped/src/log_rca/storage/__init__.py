"""Storage backends for the synthetic log corpus.

The abstract ``LogStorage`` interface lets the rest of the codebase stay
backend-agnostic. Phase 0 ships ``LocalFSBackend`` (a local directory
treated as a fake GCS bucket). A future ``GCSBackend`` would implement
the same protocol against real Google Cloud Storage.
"""

from log_rca.storage.base import LogStorage
from log_rca.storage.local import LocalFSBackend

__all__ = ["LogStorage", "LocalFSBackend"]
