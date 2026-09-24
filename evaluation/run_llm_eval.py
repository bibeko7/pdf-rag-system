from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.pipeline import RAGPipeline

PDF_PATH = PROJECT_ROOT / "data" / "evaluation" / "Machine_Learning_No_Table.pdf"
DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "test_dataset.json"
RESULTS_PATH = PROJECT_ROOT / "data" / "evaluation" / "llm_results.json"


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def main():
    dataset = load_dataset()

    completed = {}
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH, "r", encoding="utf-8-sig") as file:
                existing = json.load(file)
            for result in existing.get("results", []):
                completed[result["id"]] = result
        except (json.JSONDecodeError, OSError):
            completed = {}

    pipeline = RAGPipeline()
    pipeline.ingest(str(PDF_PATH))

    results = list(completed.values())

    for item in dataset:
        question_id = item["id"]

        if question_id in completed:
            print(f"Skipping {question_id} - already completed")
            continue

        question = item["question"]
        question_type = item["type"]

        print("=" * 80)
        print(f"Evaluating {question_id}: {question}")

        try:
            result = pipeline.ask(question)

            answer = result.answer.strip()

            record = {
                "id": question_id,
                "question": question,
                "ground_truth": item["ground_truth"],
                "type": question_type,
                "answer": answer,
                "latency_seconds": result.latency_seconds,
                "source_pages": [
                    (source.page + 1) if source.page is not None else None
                    for source in result.sources
                ],
                "status": "success",
            }

        except Exception as exc:
            record = {
                "id": question_id,
                "question": question,
                "ground_truth": item["ground_truth"],
                "type": question_type,
                "answer": "",
                "latency_seconds": None,
                "source_pages": [],
                "status": "failed",
                "error": str(exc),
            }

        results.append(record)
        completed[question_id] = record

        with open(RESULTS_PATH, "w", encoding="utf-8-sig") as file:
            json.dump(
                {
                    "results": results
                },
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Status: {record['status']}")

    print("\n" + "=" * 80)
    print("LLM evaluation completed")
    print(f"Questions: {len(results)}")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

