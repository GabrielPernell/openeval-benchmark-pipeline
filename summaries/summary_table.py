"""
summary_table.py
=================
Renders the README table for the model summary.

Separate item and response columns, because they answer different questions and
conflating them is misleading:
  items            distinct items in the benchmark (the coverage denominator)
  responses        every response in the benchmark, across all models and trials
  best_model_items most items any single model covers

Responses can far exceed items x models when models were run on the same item
more than once, so neither number can stand in for the other.

Coverage is reported as a median plus a min-max range rather than a mean.
The distribution is heavily lopsided -- on anthropic-red-teaming, 84 of 92
models sit between 1% and 9% while 6 cover ~96% -- so a mean lands in a gap
where no model actually is.
"""
import statistics


def render_table(counts: dict, totals: dict) -> str:
    lines = [
        "| benchmark | models | items | responses | best-covered model (items) "
        "| median coverage | coverage range |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, models in sorted(counts.items()):
        cov = [m["coverage"] for m in models.values() if m["coverage"] is not None]
        best = max((m["n_items"] for m in models.values()), default=0)
        total = totals.get(key, 0)
        # Counts files written before response counts existed show "-" rather
        # than a misleading 0.
        if all("responses" in m for m in models.values()):
            n_resp = sum(m["responses"] for m in models.values())
        else:
            n_resp = "-"
        if cov:
            median = round(statistics.median(cov), 1)
            rng = f"{min(cov)}%–{max(cov)}%"
        else:
            median, rng = 0, "-"
        lines.append(f"| {key} | {len(models)} | {total} | {n_resp} | {best} "
                     f"| {median}% | {rng} |")
    return chr(10).join(lines) + chr(10)
