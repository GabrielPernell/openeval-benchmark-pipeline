"""
loaders/ltb.py
==============
Adapter for the Last Translation Benchmark (arXiv 2609.04173), release LTBv1:
3,456 human-authored examples that break state-of-the-art translation models,
each with handcrafted verification rules. Data at HF
`zouhar/last-translation-benchmark` (CC BY 4.0), code at GitHub
`zouharvi/last-translation-benchmark` (MIT). Fetch with
fetchers/fetch_ltb_cache.py, which pins the HF revision.

Columns (one JSON object per example):
    id                              int    stable example id (not contiguous)
    source_text                     str    text to translate ("" for some
                                           media-only examples)
    source_lang, target_lang        str    human-readable language names
    source_lang_iso, target_lang_iso str   ISO 639-3 codes, when they exist
    source_instructions             str?   extra instructions for the translator
    source_media                    str?   base64 data URI (image/audio/video)
    attribution                     str?   resource the example was based on
    translations                    list   {model, translation, verified[]};
                                           model "human" is the reference
    verification_rules              list   rules a correct translation meets
    linguistics                     list   annotated difficulty types
    tags                            list   "LTBv1"; "LTBv1-eval" marks the 911
                                           examples the paper's leaderboard uses

Mapping, following the lab's DocHop re-upload (every source column kept under
its own name, split between input and references):
    item_content.input       [{every column except translations and
                               verification_rules}]
    item_content.references  [{"verification_rules": [...],
                               "translations": [<the human entry>]}]
    benchmark_tags           the example's own tags

The model entries of `translations` become responses (add_ltb_responses.py);
only the human one is a reference.

About the media
---------------
212 examples carry an image, audio clip or video as a base64 data URI, 64 MB in
total. As in the DocHop re-upload, the item holds a URL to the example's row
in the HF dataset viewer instead. Viewer row N is the N-th example of the
pinned v1.json (checked 2026-09-30), so the URL is built from the row's
position, not its `id`. `mime_type` is kept alongside it because the models'
prompt depended on whether the media was an image, audio or video.
"""
from adapter_base import DatasetAdapter

REPO_ID = "zouhar/last-translation-benchmark"
VIEWER_ROW_URL = f"https://huggingface.co/datasets/{REPO_ID}/viewer/default/train?row={{}}"

# Columns carried into the item input as-is; the other two are split out.
INPUT_COLUMNS = ["id", "source_text", "source_lang", "target_lang", "source_lang_iso",
                 "target_lang_iso", "source_instructions", "source_media",
                 "attribution", "linguistics", "tags"]


def media_mime(data_uri):
    """'data:audio/mpeg;base64,...' -> 'audio/mpeg'."""
    if not data_uri or not data_uri.startswith("data:") or "," not in data_uri:
        return None
    return data_uri[5:].split(",", 1)[0].split(";", 1)[0]


def human_entry(row):
    humans = [t for t in row["translations"] if t["model"] == "human"]
    if len(humans) != 1:
        raise ValueError(f"LTB example {row['id']}: {len(humans)} human translations, expected 1")
    return humans[0]


class LTBAdapter(DatasetAdapter):
    def __init__(self):
        super().__init__(
            benchmark_name="last-translation-benchmark",
            benchmark_version="LTBv1",
            paper_url="https://arxiv.org/abs/2609.04173",
            dataset_url=f"https://huggingface.co/datasets/{REPO_ID}",
            base_tags=["translation", "multilingual", "multimodal", "verification-rules"],
        )

    def load_rows(self, source: str, limit: int | None = None, mode: str = "cache"):
        import os as _os, sys as _sys
        _fetchers = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "fetchers")
        if _fetchers not in _sys.path:
            _sys.path.insert(0, _fetchers)
        import fetch_ltb_cache

        if mode == "hf":
            source = fetch_ltb_cache.fetch(source or "data_cache/ltb/v1.json")
        elif mode != "cache":
            raise ValueError(f"Unknown mode '{mode}', expected 'cache' or 'hf'")

        rows = fetch_ltb_cache.load(source)
        # The viewer URL needs each example's position in the file.
        for i, row in enumerate(rows):
            row["_row_index"] = i
        return rows[:limit] if limit else rows

    def _input(self, row):
        if not row["source_text"].strip() and not row["source_media"]:
            raise ValueError(f"LTB example {row['id']} has neither text nor media")
        payload = {col: row[col] for col in INPUT_COLUMNS}
        if row["source_media"]:
            payload["source_media"] = {
                "url": VIEWER_ROW_URL.format(row["_row_index"]),
                "mime_type": media_mime(row["source_media"]),
            }
        return payload

    def _references(self, row):
        if not row["verification_rules"]:
            raise ValueError(f"LTB example {row['id']} has no verification rules")
        return {
            "verification_rules": list(row["verification_rules"]),
            "translations": [human_entry(row)],
        }

    def row_to_content(self, row: dict) -> dict:
        return {
            "input": [self._input(row)],
            "references": [self._references(row)],
            "tags": list(row["tags"]),
        }

    def repo_input(self, row: dict) -> dict:
        return self._input(row)

    def repo_references(self, row: dict) -> list:
        return [self._references(row)]

    def row_uid(self, row: dict) -> str:
        return str(row["id"])
