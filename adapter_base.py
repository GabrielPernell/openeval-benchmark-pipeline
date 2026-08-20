"""
adapter_base.py
================
Defines the interface every per-dataset loader must implement.

To add a new benchmark to this pipeline (DocHop, VisionWebDev, or anything else):

    1. Create loaders/<your_dataset>.py
    2. Subclass DatasetAdapter and implement:
         - load_rows(source, limit=None)  -> yields raw rows (dicts)
         - row_to_content(row)            -> {"input": [...], "references": [...], "tags": [...]}
         - row_uid(row)                   -> a stable string id for the row (for logging/dedup)
    3. Fill in the class-level metadata fields (benchmark_name, benchmark_version, ...)
    4. Register an instance of it in convert_to_openeval.py's ADAPTERS dict.

That's the whole contract. Everything else (item_id generation, assembling the
full item dict, writing JSON, running the validator) is handled generically by
schema_utils.py + convert_to_openeval.py.
"""
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class DatasetAdapter:
    # --- item_metadata.source fields (shared by every item from this benchmark) ---
    benchmark_name: str
    benchmark_version: str
    paper_url: str = ""
    dataset_url: str = ""
    base_tags: list[str] = field(default_factory=list)

    def load_rows(self, source: str, limit: int | None = None) -> Iterable[dict]:
        """Yield raw rows from `source` (a local cache path, a HF dataset id, etc.)."""
        raise NotImplementedError

    def row_to_content(self, row: dict) -> dict:
        """Map one raw row -> {"input": list[str|dict], "references": list[str|dict], "tags": list[str]}."""
        raise NotImplementedError

    def row_uid(self, row: dict) -> str:
        """Return a stable identifier for `row`, used only for logging / de-duplication."""
        raise NotImplementedError
