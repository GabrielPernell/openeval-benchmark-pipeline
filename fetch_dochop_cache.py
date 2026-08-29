#!/usr/bin/env python3
"""
fetch_dochop_cache.py
======================
Build a local cache file for loaders/dochop.py.

Why this exists
---------------
DocHop ships as a single 2.4 GB TSV (`DocHop.tsv`) whose `image` column holds
a base64-encoded PNG of a full document page -- roughly 755 KB per row across
2,074 rows. Embedding those in the converted items would produce a ~1.5 GB
JSON file, so by default this script keeps only the metadata columns and the
items reference each image by its row position instead.

HuggingFace auto-converts the dataset to parquet, which is columnar, so the
metadata columns can be read with range requests in a few seconds without
transferring the image data at all.

Pass --images-dir to additionally materialise the PNGs on disk (this DOES
stream the full ~2.4 GB and writes ~1.5 GB of files); the cache then records
a relative path per row and the converted items point at real files, the same
way the VisionWebDev items point into an extracted archive.

Usage
-----
python fetch_dochop_cache.py --out data_cache/dochop_raw.json
python fetch_dochop_cache.py --out data_cache/dochop_raw.json --images-dir data_cache/dochop_images
"""
import argparse
import base64
import json
import os
import sys

REPO_ID = "zhuoranyu336/dochop"
PARQUET_URL = (
    "https://huggingface.co/datasets/zhuoranyu336/dochop/resolve/"
    "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
META_COLUMNS = ["index", "original_id", "question", "answer", "task", "chart_num", "depth"]


def _open_parquet():
    import fsspec
    import pyarrow.parquet as pq
    return pq.ParquetFile(fsspec.filesystem("http").open(PARQUET_URL, "rb"))


def fetch_rows(limit=None, images_dir=None):
    pf = _open_parquet()
    print(f"      -> {pf.metadata.num_rows} rows, {pf.metadata.num_row_groups} row groups",
          file=sys.stderr)

    rows = [dict(r) for r in pf.read(columns=META_COLUMNS).to_pylist()]
    if limit:
        rows = rows[:limit]
    print(f"      -> read metadata for {len(rows)} rows (image column skipped)", file=sys.stderr)

    if images_dir:
        os.makedirs(images_dir, exist_ok=True)
        by_index = {r["index"]: r for r in rows}
        written = 0
        # Re-read including the image column, one row group at a time so the
        # base64 blobs are never all in memory at once.
        for batch in pf.iter_batches(batch_size=64, columns=["index", "image"]):
            for rec in batch.to_pylist():
                target = by_index.get(rec["index"])
                if target is None:
                    continue
                name = f"{target['original_id']}.png"
                with open(os.path.join(images_dir, name), "wb") as f:
                    f.write(base64.b64decode(rec["image"]))
                target["image_path"] = f"{os.path.basename(images_dir)}/{name}"
                written += 1
            if limit and written >= len(rows):
                break
            if written and written % 256 == 0:
                print(f"      ... wrote {written}/{len(rows)} images", file=sys.stderr)
        print(f"      -> wrote {written} PNG(s) to {images_dir}", file=sys.stderr)

    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--images-dir", default=None,
                    help="Also decode each row's PNG to this directory "
                         "(streams ~2.4 GB, writes ~1.5 GB)")
    args = ap.parse_args()

    print(f"[1/2] Reading DocHop metadata from the parquet conversion of {REPO_ID}...")
    rows = fetch_rows(limit=args.limit, images_dir=args.images_dir)

    missing = [r["original_id"] for r in rows
               if not str(r.get("question") or "").strip() or not str(r.get("answer") or "").strip()]
    if missing:
        print(f"      ! WARNING: {len(missing)} row(s) missing question or answer: {missing[:5]}",
              file=sys.stderr)

    print(f"[2/2] Writing {len(rows)} rows to {args.out}...")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"      -> {os.path.getsize(args.out) / 1e6:.1f} MB cache written")


if __name__ == "__main__":
    main()
