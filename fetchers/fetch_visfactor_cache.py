#!/usr/bin/env python3
"""
fetch_visfactor_cache.py
========================
Build the local caches loaders/visfactor.py and add_visfactor_responses.py use.

Items
-----
VisFactor is one parquet file on HF (`lmms-lab-encoder/visfactor`,
`test.parquet`, 3,046 rows, 76 MB). The `image` column holds 1-3 embedded
images per row; the items reference them by viewer-row URL instead, so only
the other columns are kept, plus how many images each row has. The download is
pinned to one HF revision and checked against its sha256: the viewer-row URLs
are only valid for this file's row order.

Responses
---------
The model responses are not on HF. The lab shared them privately as
`VisFactor.zip` on Google Drive (one VLMEvalKit result file per model,
standard prompt only), so the file's Drive id is not in this repository: set
VISFACTOR_DRIVE_FILE_ID to it. `--responses` downloads and extracts the zip;
the files land in data_cache/visfactor/Vanilla/.

Usage
-----
python fetchers/fetch_visfactor_cache.py
VISFACTOR_DRIVE_FILE_ID=<id> python fetchers/fetch_visfactor_cache.py --responses
"""
import argparse
import hashlib
import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net_utils import get_with_retry

REPO_ID = "lmms-lab-encoder/visfactor"
REVISION = "bab7178b7b86b24a0713d4ec48447209e749203c"
PATH = "test.parquet"
SHA256 = "5408f4296ec01883c3143714b6980f03d0cfe5371c2ad98975c2988e844bfac6"
N_ROWS = 3046
URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{PATH}"

# Shared privately by the lab; not published here.
DRIVE_FILE_ID = os.environ.get("VISFACTOR_DRIVE_FILE_ID", "")
DRIVE_DOWNLOAD_URL = (f"https://drive.usercontent.google.com/download?id={DRIVE_FILE_ID}"
                      "&export=download&confirm=t")
RESPONSES_SUBDIR = "Vanilla"

META_COLUMNS = ["index", "category_name", "category_id", "eval_index",
                "question", "answer", "additional"]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_to(url, out, headers=None):
    resp = get_with_retry(url, stream=True, timeout=120, headers=headers)
    tmp = out + ".part"
    with open(tmp, "wb") as f:
        for chunk in resp.iter_content(1 << 20):
            f.write(chunk)
    return tmp


def fetch_items(cache_dir):
    """Download the pinned parquet and write <cache_dir>/visfactor_items.json."""
    import pyarrow.parquet as pq

    os.makedirs(cache_dir, exist_ok=True)
    parquet = os.path.join(cache_dir, PATH)
    if not (os.path.exists(parquet) and sha256_of(parquet) == SHA256):
        tmp = stream_to(URL, parquet)
        got = sha256_of(tmp)
        if got != SHA256:
            os.remove(tmp)
            raise RuntimeError(f"sha256 mismatch for {URL}: got {got}, expected {SHA256}")
        os.replace(tmp, parquet)
    print(f"      -> {parquet} matches revision {REVISION[:8]}")

    table = pq.read_table(parquet)
    rows = table.select(META_COLUMNS).to_pylist()
    for row, images in zip(rows, table.column("image").to_pylist()):
        row["n_images"] = len(images)
    # The viewer-row URLs assume `index` is the row's position in the file.
    if [r["index"] for r in rows] != list(range(len(rows))) or len(rows) != N_ROWS:
        raise ValueError(f"{parquet}: rows are not indexed 0..{N_ROWS - 1}")

    out = os.path.join(cache_dir, "visfactor_items.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"      -> {len(rows)} rows written to {out}")
    return out


def fetch_responses(cache_dir):
    """Download VisFactor.zip from the lab's Drive link and extract it."""
    os.makedirs(cache_dir, exist_ok=True)
    zpath = os.path.join(cache_dir, "VisFactor.zip")
    if not os.path.exists(zpath):
        if not DRIVE_FILE_ID:
            raise SystemExit("Set VISFACTOR_DRIVE_FILE_ID to the Drive id of the lab's "
                             f"VisFactor.zip, or place the zip at {zpath}.")
        # Not HuggingFace: no HF_TOKEN.
        os.replace(stream_to(DRIVE_DOWNLOAD_URL, zpath, headers={}), zpath)
    with zipfile.ZipFile(zpath) as z:
        members = [m for m in z.namelist() if not m.startswith("__MACOSX")]
        z.extractall(cache_dir, members=members)
    out = os.path.join(cache_dir, RESPONSES_SUBDIR)
    print(f"      -> {len(os.listdir(out))} files in {out}")
    return out


def load_items(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    if len(rows) != N_ROWS:
        raise ValueError(f"{path}: {len(rows)} rows, expected {N_ROWS}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="data_cache/visfactor")
    ap.add_argument("--responses", action="store_true",
                    help="also download and extract the lab's VisFactor.zip (~200 MB)")
    args = ap.parse_args()

    print(f"[1/2] Fetching {REPO_ID} {PATH} at revision {REVISION[:8]}...")
    fetch_items(args.cache_dir)
    if args.responses:
        print("[2/2] Fetching the model responses from Google Drive...")
        fetch_responses(args.cache_dir)


if __name__ == "__main__":
    main()
