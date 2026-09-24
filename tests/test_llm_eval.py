import pytest

from evaluation.llm_eval import (
    answer_length,
    contains_answer,
    exact_match,
    evaluate_answer,
    is_refusal_answer,
)


def test_exact_match():
    assert exact_match(
        "Machine Learning",
        "Machine Learning",
    ) == 1.0


def test_exact_match_is_case_insensitive():
    assert exact_match(
        "Machine Learning",
        "machine learning",
    ) == 1.0


def test_exact_match_rejects_different_answers():
    assert exact_match(
        "Deep Learning",
        "Machine Learning",
    ) == 0.0


def test_contains_answer():
    assert contains_answer(
        "Machine Learning is a subset of AI.",
        "Machine Learning",
    ) == 1.0


def test_contains_answer_rejects_missing_answer():
    assert contains_answer(
        "Deep Learning is a subset of AI.",
        "Machine Learning",
    ) == 0.0


def test_answer_length():
    assert answer_length(" hello ") == 5


def test_answer_length_empty():
    assert answer_length("") == 0


def test_refusal_answer():
    assert is_refusal_answer(
        "I could not find the answer in the provided document."
    )


def test_non_refusal_answer():
    assert not is_refusal_answer(
        "Machine Learning is a field of artificial intelligence."
    )


def test_answerable_evaluation():
    result = evaluate_answer(
        prediction="Machine Learning is a field of AI.",
        ground_truth="Machine Learning",
        question_type="answerable",
    )

    assert result["contains_answer"] == 1.0
    assert result["correct_refusal"] == 0.0


def test_out_of_context_evaluation():
    result = evaluate_answer(
        prediction=(
            "I could not find the answer in the provided document."
        ),
        ground_truth="",
        question_type="out_of_context",
    )

    assert result["correct_refusal"] == 1.0
    assert result["exact_match"] == 0.0


def test_none_prediction():
    assert exact_match(None, "test") == 0.0
    assert contains_answer(None, "test") == 0.0
    assert answer_length(None) == 0


def test_invalid_empty_ground_truth():
    assert contains_answer(
        "some answer",
        "",
    ) == 0.0
