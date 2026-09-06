"""
summary_table.py
=================
Renders the README table for the model summary.

Two separate item columns, because they answer different questions and
conflating them is misleading:
  items            distinct items in the benchmark (the coverage denominator)
  best_model_items most items any single model covers

Coverage is reported as a median plus a min-max range rather than a mean.
The distribution is heavily lopsided -- on anthropic-red-teaming, 84 of 92
models sit between 1% and 9% while 6 cover ~96% -- so a mean lands in a gap
where no model actually is.
"""
import statistics


def render_table(counts: dict, totals: dict) -> str:
    lines = [
        "| benchmark | models | items | best-covered model (items) | median coverage | coverage range |",
        "|---|---|---|---|---|---|",
    ]
    for key, models in sorted(counts.items()):
        cov = [m["coverage"] for m in models.values() if m["coverage"] is not None]
        best = max((m["n_items"] for m in models.values()), default=0)
        total = totals.get(key, 0)
        if cov:
            median = round(statistics.median(cov), 1)
            rng = f"{min(cov)}%–{max(cov)}%"
        else:
            median, rng = 0, "-"
        lines.append(f"| {key} | {len(models)} | {total} | {best} | {median}% | {rng} |")
    return chr(10).join(lines) + chr(10)
