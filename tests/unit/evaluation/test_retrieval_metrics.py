import pytest

from rag.evaluation.metrics import (
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k():
    assert precision_at_k(["a", "x", "b", "y"], {"a", "b"}, 3) == pytest.approx(2 / 3)


def test_recall_at_k():
    assert recall_at_k(["x", "a", "b"], {"a", "b", "c"}, 2) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_first_relevant_result():
    assert reciprocal_rank(["x", "y", "relevant", "also-relevant"], {"relevant"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_no_relevant_result():
    assert reciprocal_rank(["x", "y"], {"relevant"}) == 0.0


def test_mean_reciprocal_rank():
    runs = [["a", "x"], ["x", "b"], ["x", "y"]]
    relevant = [{"a"}, {"b"}, {"z"}]
    assert mean_reciprocal_rank(runs, relevant) == pytest.approx((1 + 0.5 + 0) / 3)


def test_mean_reciprocal_rank_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="Each retrieval run"):
        mean_reciprocal_rank([["a"]], [])
