def exact_match(prediction: str, ground_truth: str) -> float:
    if prediction is None or ground_truth is None:
        return 0.0

    prediction = prediction.strip().lower()
    ground_truth = ground_truth.strip().lower()

    if not prediction or not ground_truth:
        return 0.0

    return float(prediction == ground_truth)


def contains_answer(prediction: str, ground_truth: str) -> float:
    if prediction is None or ground_truth is None:
        return 0.0

    prediction = prediction.strip().lower()
    ground_truth = ground_truth.strip().lower()

    if not prediction or not ground_truth:
        return 0.0

    return float(ground_truth in prediction)


def answer_length(prediction: str) -> int:
    if not prediction:
        return 0

    return len(prediction.strip())


def is_refusal_answer(answer: str) -> bool:
    if not answer:
        return False

    normalized = answer.strip().lower()

    refusal_phrases = [
        "i could not find the answer in the provided document",
        "i couldn't find the answer in the provided document",
        "the answer cannot be found in the provided document",
        "not found in the provided document",
    ]

    return any(
        phrase in normalized
        for phrase in refusal_phrases
    )


def evaluate_answer(
    prediction: str,
    ground_truth: str,
    question_type: str,
):
    if question_type == "out_of_context":
        return {
            "exact_match": 0.0,
            "contains_answer": 0.0,
            "answer_length": answer_length(prediction),
            "correct_refusal": float(is_refusal_answer(prediction)),
        }

    return {
        "exact_match": exact_match(
            prediction,
            ground_truth,
        ),
        "contains_answer": contains_answer(
            prediction,
            ground_truth,
        ),
        "answer_length": answer_length(prediction),
        "correct_refusal": 0.0,
    }
