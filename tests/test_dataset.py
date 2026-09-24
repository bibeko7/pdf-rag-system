import pytest

from evaluation.dataset import load_evaluation_dataset


def test_load_evaluation_dataset():
    data = load_evaluation_dataset()

    assert isinstance(data, list)
    assert len(data) == 15


def test_evaluation_dataset_structure():
    data = load_evaluation_dataset()

    for item in data:
        assert "id" in item
        assert "question" in item
        assert "ground_truth" in item
        assert "relevant_pages" in item
        assert "type" in item

        assert item["id"]
        assert item["question"]
        assert isinstance(item["relevant_pages"], list)
        assert item["type"] in {"answerable", "out_of_context"}


def test_answerable_questions_have_ground_truth():
    data = load_evaluation_dataset()

    answerable = [
        item for item in data
        if item["type"] == "answerable"
    ]

    assert len(answerable) == 14

    for item in answerable:
        assert item["ground_truth"].strip()
        assert item["relevant_pages"]


def test_out_of_context_questions():
    data = load_evaluation_dataset()

    out_of_context = [
        item for item in data
        if item["type"] == "out_of_context"
    ]

    assert len(out_of_context) == 1

    for item in out_of_context:
        assert item["ground_truth"] == ""
        assert item["relevant_pages"] == []
