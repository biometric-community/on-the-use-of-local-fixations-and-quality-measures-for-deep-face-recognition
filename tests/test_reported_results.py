"""REPORT.md, the committed output of the full run, against the paper's premise."""

from __future__ import annotations

import re

import pytest


def row(report: str, dataset: str) -> tuple[float, float]:
    match = re.search(rf"\| {dataset} \| ([\d.]+)% \| [\d.]+% \| ([\d.]+)% \|", report)
    assert match, f"{dataset} row missing from REPORT.md"
    return float(match.group(1)), float(match.group(2))


@pytest.mark.parametrize("dataset", ["lfw", "calfw", "cplfw"])
def test_quality_weighted_matching_does_not_lose_to_global(report, targets, dataset):
    """Quality-driven local matching is the proposal, so it should not fall behind.

    This is the method's stated purpose rather than a published table.
    """
    global_acc, weighted_acc = row(report, dataset)
    assert weighted_acc >= global_acc - targets["noise_pp"]


def test_published_values(report, targets):
    if targets["published_values"] is None:
        pytest.skip("paper tables not recorded in tests/paper_targets.yaml")
    for dataset, expected in targets["published_values"].items():
        global_acc, weighted_acc = row(report, dataset)
        assert global_acc == pytest.approx(expected["global"], abs=1.0)
        assert weighted_acc == pytest.approx(expected["quality_weighted"], abs=1.0)
