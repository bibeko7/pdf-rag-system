import pytest

from evaluation.retriever_eval import (
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    evaluate_retrieval,
)


def test_hit_at_k():
    assert hit_at_k([3, 5, 2, 7], [3], 4) == 1.0
    assert hit_at_k([5, 2, 7, 8], [3], 4) == 0.0


def test_precision_at_k():
    assert precision_at_k([3, 5, 2, 7], [3], 4) == 0.25
    assert precision_at_k([3, 5, 2, 7], [3, 5], 4) == 0.5


def test_recall_at_k():
    assert recall_at_k([3, 5, 2, 7], [3], 4) == 1.0
    assert recall_at_k([3, 5, 2, 7], [3, 5], 4) == 1.0
    assert recall_at_k([5, 2, 7, 8], [3, 5], 4) == 0.5


def test_reciprocal_rank():
    assert reciprocal_rank([3, 5, 2, 7], [3]) == 1.0
    assert reciprocal_rank([5, 3, 2, 7], [3]) == 0.5
    assert reciprocal_rank([5, 2, 7, 3], [3]) == 0.25
    assert reciprocal_rank([5, 2, 7, 8], [3]) == 0.0


def test_mean_reciprocal_rank():
    results = [
        {
            "retrieved_pages": [3, 5, 2, 7],
            "relevant_pages": [3],
        },
        {
            "retrieved_pages": [5, 3, 2, 7],
            "relevant_pages": [3],
        },
    ]

    assert mean_reciprocal_rank(results) == 0.75


def test_evaluate_retrieval():
    result = evaluate_retrieval(
        retrieved_pages=[3, 5, 2, 7],
        relevant_pages=[3],
        k=4,
    )

    assert result["hit@4"] == 1.0
    assert result["precision@4"] == 0.25
    assert result["recall@4"] == 1.0
    assert result["reciprocal_rank"] == 1.0


def test_invalid_k():
    with pytest.raises(ValueError):
        hit_at_k([1, 2], [1], 0)

    with pytest.raises(ValueError):
        precision_at_k([1, 2], [1], 0)

    with pytest.raises(ValueError):
        recall_at_k([1, 2], [1], 0)


def test_empty_retrieval():
    assert hit_at_k([], [1], 4) == 0.0
    assert precision_at_k([], [1], 4) == 0.0
    assert recall_at_k([], [1], 4) == 0.0
    assert reciprocal_rank([], [1]) == 0.0
