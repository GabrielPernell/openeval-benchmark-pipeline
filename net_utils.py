"""
net_utils.py
=============
Shared HTTP helpers for the loaders. Factored out of loaders/dsr_bench.py once
loaders/visionwebdev.py needed the same rate-limit handling.

HuggingFace's datasets-server rate-limits unauthenticated clients, and a full
subset fetch is hundreds of sequential requests, so every caller needs pacing
plus backoff rather than dying and losing the run.
"""
import os
import random
import sys
import time

PAGE_DELAY_S = 0.5
MAX_RETRIES = 6
RETRYABLE = {429, 500, 502, 503, 504}

DATASETS_SERVER_ROWS = "https://datasets-server.huggingface.co/rows"


def hf_headers() -> dict:
    """Authorization header if HF_TOKEN is set in the environment.

    Authenticated clients get a substantially higher rate limit. The token is
    read from the environment only -- never logged, never written to output.
    """
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def get_with_retry(url: str, params: dict | None = None, timeout: int = 30, stream: bool = False):
    """GET with exponential backoff on rate-limit/transient errors."""
    import requests

    delay = 2.0
    last_status = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=hf_headers(),
                                timeout=timeout, stream=stream)
        except requests.RequestException as exc:
            # Timeouts and dropped connections are as transient as a 503 and
            # must not kill a long paginated fetch.
            if attempt == MAX_RETRIES - 1:
                raise
            wait = delay + random.uniform(0, 1)
            print(f"      ! {type(exc).__name__} -- retry {attempt + 1}/{MAX_RETRIES - 1} "
                  f"in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
            delay = min(delay * 2, 60.0)
            continue
        if resp.status_code == 200:
            return resp

        last_status = resp.status_code
        if resp.status_code not in RETRYABLE:
            hint = ""
            if resp.status_code in (401, 403):
                hint = (
                    "\nA 403 here usually means an outbound proxy is blocking huggingface.co "
                    "(some sandboxes do) -- fall back to --mode cache with a pre-fetched JSON "
                    "file (see data_cache/ for the format)."
                )
            raise RuntimeError(f"HF request failed ({resp.status_code}): {resp.text[:300]}{hint}")

        try:
            wait = float(resp.headers.get("Retry-After", ""))
        except ValueError:
            wait = delay
        wait = max(wait, delay) + random.uniform(0, 1)
        print(
            f"      ! HTTP {resp.status_code} (rate limit / transient) -- retry "
            f"{attempt + 1}/{MAX_RETRIES} in {wait:.1f}s",
            file=sys.stderr,
        )
        time.sleep(wait)
        delay = min(delay * 2, 60.0)

    raise RuntimeError(
        f"HF endpoint still returning {last_status} after {MAX_RETRIES} retries. You are being "
        "rate-limited harder than the backoff can absorb. Options: wait a few minutes and re-run, "
        "or set an HF_TOKEN environment variable (authenticated requests get a much higher limit)."
    )


def fetch_hf_rows(repo_id: str, config: str, split: str = "test", limit: int | None = None,
                  page_size: int = 100) -> list[dict]:
    """Page through datasets-server /rows and return the raw row dicts.

    Shared by every loader that reads a tabular HF dataset.
    """
    rows: list[dict] = []
    offset = 0
    total = None
    while True:
        resp = get_with_retry(
            DATASETS_SERVER_ROWS,
            {"dataset": repo_id, "config": config, "split": split,
             "offset": offset, "length": page_size},
        )
        data = resp.json()
        page_rows = [r["row"] for r in data["rows"]]
        rows.extend(page_rows)
        offset += page_size

        total = data["num_rows_total"]
        target = min(limit, total) if limit else total
        # Progress on stderr so it stays visible when stdout is redirected.
        print(f"      ... fetched {min(len(rows), target)}/{target} rows", file=sys.stderr)

        if limit and len(rows) >= limit:
            return rows[:limit]
        if offset >= total:
            break
        if not page_rows:
            print(
                f"      ! empty page at offset {offset}; stopping with {len(rows)} rows "
                f"(expected {total})",
                file=sys.stderr,
            )
            break
        time.sleep(PAGE_DELAY_S)

    if total is not None and not limit and len(rows) != total:
        print(
            f"      ! WARNING: fetched {len(rows)} rows but the server reports {total} "
            "-- output is incomplete.",
            file=sys.stderr,
        )
    return rows
