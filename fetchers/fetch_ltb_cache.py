#!/usr/bin/env python3
"""
fetch_ltb_cache.py
==================
Download the Last Translation Benchmark release for loaders/ltb.py.

LTB is a single JSON file on HF (`zouhar/last-translation-benchmark`,
`data/v1.json`, ~90 MB, 3,456 examples). It is a *live* dataset -- the authors
plan further releases -- so the download is pinned to one HF revision and
checked against that revision's sha256, and the loader's viewer-row URLs are
only valid for this file's row order.

The model responses live in the same file (each example's `translations`
list), so this one download is everything the conversion needs. The file is
cached as-is, base64 media included: the loader replaces media with a URL, but
the response step needs the media type to rebuild each model's prompt.

Usage
-----
python fetchers/fetch_ltb_cache.py --out data_cache/ltb/v1.json
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net_utils import get_with_retry

REPO_ID = "zouhar/last-translation-benchmark"
# LTBv1 as released on 2026-09-04. Bump all three together for a new release.
REVISION = "a483825ddbe2d7756f5bdfb1e4f611bee9026c4c"
PATH = "data/v1.json"
SHA256 = "99fa3d547530b64b45fa3181811fba11281a013c53b9959091f770a5f6594b9f"
N_EXAMPLES = 3456

URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{PATH}"


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(out):
    """Download the pinned file to `out` unless an identical copy is there."""
    if os.path.exists(out) and sha256_of(out) == SHA256:
        print(f"      -> {out} already matches revision {REVISION[:8]}")
        return out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    resp = get_with_retry(URL, stream=True, timeout=120)
    tmp = out + ".part"
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(1 << 20):
            f.write(chunk)
    got = sha256_of(tmp)
    if got != SHA256:
        os.remove(tmp)
        raise RuntimeError(f"sha256 mismatch for {URL}: got {got}, expected {SHA256}")
    os.replace(tmp, out)
    print(f"      -> {os.path.getsize(out) / 1e6:.1f} MB written to {out}")
    return out


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    if len(rows) != N_EXAMPLES:
        raise ValueError(f"{path}: {len(rows)} examples, expected {N_EXAMPLES}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_cache/ltb/v1.json")
    args = ap.parse_args()

    print(f"[1/2] Fetching {REPO_ID} {PATH} at revision {REVISION[:8]}...")
    fetch(args.out)

    print("[2/2] Checking contents...")
    rows = load(args.out)
    n_trans = sum(len(r["translations"]) for r in rows)
    n_media = sum(bool(r["source_media"]) for r in rows)
    print(f"      -> {len(rows)} examples, {n_trans} translations, {n_media} with media")


if __name__ == "__main__":
    main()
