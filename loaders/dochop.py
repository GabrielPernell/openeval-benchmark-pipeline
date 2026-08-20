"""
loaders/dochop.py
===================
Adapter scaffold for DocHop ("DocHop: Benchmarking Out-of-domain Multi-hop
Reasoning in Information-Dense Documents", Zhuoran Yu et al., ICML 2026 /
PMLR vol. 306).

STATUS (checked 2026-08-15): NOT YET USABLE. Per the availability check in
the project doc `claude/dataset-availability-check.md`, the paper's project
page (zhuoranyu.com/dochop-page) has non-functional "Dataset" and "Code"
links, and no HuggingFace/GitHub release could be found despite the paper
being publicly published. This class is a scaffold only -- it documents the
mapping we *would* use once the authors release the data, so that plugging
this dataset into the pipeline later is a matter of filling in load_rows().

Expected item_content mapping once data is available (from reading the paper,
Section 3 and Appendix D):
    - item_content.input: the rendered single-page document image (chart(s)
      + narrative text) -- each DocHop instance is one document-page image,
      so `input` would likely be a single dict like
      {"image": <path-or-url-or-base64>, "modality": "image"}, or possibly
      the narrative text extracted alongside the image, depending on what
      the release actually contains.
    - item_content.references: the ground-truth answer to the grounded
      question (computed deterministically from the underlying chart data
      tables per the paper's construction pipeline), plus optionally the
      chart data tables themselves if released as structured data.
    - Per-item tags: task category (Value Retrieval / Counting / Numeric
      Reasoning / Ranking / Hypothetical Reasoning / Fact Checking -- see
      paper Section 3.3), reasoning depth, and chart count are all natural
      candidates for item_metadata.source.benchmark_tags.
"""
from adapter_base import DatasetAdapter


class DocHopAdapter(DatasetAdapter):
    def __init__(self):
        super().__init__(
            benchmark_name="DocHop",
            benchmark_version="",  # TODO: fill in once a versioned release exists
            paper_url="https://openreview.net/forum?id=PQFkScoGqz",
            dataset_url="",  # TODO: fill in once the authors publish a working dataset link
            base_tags=["reasoning", "multi-hop", "chart-qa", "document-understanding", "synthetic"],
        )

    def load_rows(self, source: str, limit: int | None = None, mode: str = "cache"):
        raise NotImplementedError(
            "DocHop's dataset is not publicly released yet (project page's Dataset/Code "
            "links are placeholders, no HuggingFace/GitHub repo found as of 2026-08-15). "
            "Reach out to the corresponding author listed on the paper for "
            "data access, then implement this method the same way loaders/dsr_bench.py "
            "does (either a live-network fetch or a local JSON/image-directory cache)."
        )

    def row_to_content(self, row: dict) -> dict:
        raise NotImplementedError("Implement once DocHop's actual released row/file format is known.")

    def row_uid(self, row: dict) -> str:
        raise NotImplementedError("Implement once DocHop's actual released row/file format is known.")
