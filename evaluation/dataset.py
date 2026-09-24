import json
from pathlib import Path


REQUIRED_FIELDS = {
    "id",
    "question",
    "ground_truth",
    "relevant_pages",
    "type",
}

VALID_TYPES = {
    "answerable",
    "out_of_context",
}


def load_evaluation_dataset(path: str = "data/evaluation/test_dataset.json"):
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}"
        )

    with dataset_path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Evaluation dataset must be a JSON list.")

    if not data:
        raise ValueError("Evaluation dataset cannot be empty.")

    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"Evaluation item {index} must be a JSON object."
            )

        missing_fields = REQUIRED_FIELDS - item.keys()

        if missing_fields:
            raise ValueError(
                f"Evaluation item {index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        if not item["id"]:
            raise ValueError(
                f"Evaluation item {index} has an empty id."
            )

        if not item["question"].strip():
            raise ValueError(
                f"Evaluation item {index} has an empty question."
            )

        if item["type"] not in VALID_TYPES:
            raise ValueError(
                f"Evaluation item {index} has invalid type: "
                f"{item['type']}"
            )

        if not isinstance(item["relevant_pages"], list):
            raise ValueError(
                f"Evaluation item {index}: relevant_pages must be a list."
            )

        if item["type"] == "answerable" and not item["ground_truth"].strip():
            raise ValueError(
                f"Evaluation item {index}: answerable question "
                "must have a ground truth."
            )

        if item["type"] == "out_of_context":
            if item["ground_truth"] != "":
                raise ValueError(
                    f"Evaluation item {index}: out_of_context question "
                    "must have an empty ground_truth."
                )

            if item["relevant_pages"]:
                raise ValueError(
                    f"Evaluation item {index}: out_of_context question "
                    "must have no relevant pages."
                )

    return data
