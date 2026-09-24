from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RETRIEVER_RESULTS_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "retriever_results.json"
)

LLM_RESULTS_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "llm_results.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "pipeline_results.json"
)

REFUSAL_TEXT = "I could not find the answer in the provided document."


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def is_refusal(answer):
    if not answer:
        return False

    return REFUSAL_TEXT.lower() in answer.strip().lower()


def main():
    retriever_data = load_json(RETRIEVER_RESULTS_PATH)
    llm_data = load_json(LLM_RESULTS_PATH)

    retriever_questions = {
        item["id"]: item
        for item in retriever_data["questions"]
    }

    llm_questions = {
        item["id"]: item
        for item in llm_data["results"]
    }

    common_ids = [
        question_id
        for question_id in retriever_questions
        if question_id in llm_questions
    ]

    if not common_ids:
        raise ValueError("No matching evaluation questions found.")

    results = []

    for question_id in common_ids:
        retriever = retriever_questions[question_id]
        llm = llm_questions[question_id]

        question_type = llm["type"]
        answer = llm.get("answer", "")
        refusal = is_refusal(answer)

        if question_type == "answerable":
            behavior_correct = not refusal
        elif question_type == "out_of_context":
            behavior_correct = refusal
        else:
            behavior_correct = False

        results.append(
            {
                "id": question_id,
                "question": llm["question"],
                "type": question_type,
                "answer": answer,
                "ground_truth": llm.get("ground_truth", ""),
                "retrieved_pages": retriever["retrieved_pages"],
                "relevant_pages": retriever["relevant_pages"],
                "hit_at_4": retriever["hit@4"],
                "precision_at_4": retriever["precision@4"],
                "recall_at_4": retriever["recall@4"],
                "reciprocal_rank": retriever["reciprocal_rank"],
                "source_pages": llm.get("source_pages", []),
                "refusal": refusal,
                "answer_behavior_correct": behavior_correct,
                "latency_seconds": llm.get("latency_seconds"),
                "status": llm.get("status", "unknown"),
            }
        )

    successful_results = [
        result
        for result in results
        if result["status"] == "success"
        and result["latency_seconds"] is not None
    ]

    total = len(results)

    answer_behavior_accuracy = (
        sum(result["answer_behavior_correct"] for result in results) / total
    )

    retrieval_hit_at_4 = (
        sum(result["hit_at_4"] for result in results) / total
    )

    if successful_results:
        average_latency = (
            sum(result["latency_seconds"] for result in successful_results)
            / len(successful_results)
        )
    else:
        average_latency = None

    pipeline_results = {
        "evaluation": "pipeline",
        "retriever_source": str(RETRIEVER_RESULTS_PATH.relative_to(PROJECT_ROOT)),
        "llm_source": str(LLM_RESULTS_PATH.relative_to(PROJECT_ROOT)),
        "summary": {
            "questions_evaluated": total,
            "successful_questions": len(successful_results),
            "answer_behavior_accuracy": answer_behavior_accuracy,
            "retrieval_hit_at_4": retrieval_hit_at_4,
            "average_latency_seconds": average_latency,
        },
        "results": results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            pipeline_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    summary = pipeline_results["summary"]

    print("=" * 80)
    print("PIPELINE EVALUATION RESULTS")
    print("=" * 80)
    print(f"Questions evaluated: {summary['questions_evaluated']}")
    print(f"Successful questions: {summary['successful_questions']}")
    print(
        f"Answer behavior accuracy: "
        f"{summary['answer_behavior_accuracy']:.4f}"
    )
    print(
        f"Retrieval Hit@4: "
        f"{summary['retrieval_hit_at_4']:.4f}"
    )

    if summary["average_latency_seconds"] is not None:
        print(
            f"Average latency: "
            f"{summary['average_latency_seconds']:.3f} seconds"
        )

    print()
    print(f"Results saved to: {OUTPUT_PATH}")

    print()
    print("=" * 80)
    print("PER-QUESTION RESULTS")
    print("=" * 80)

    for result in results:
        print(
            f"{result['id']} | "
            f"Hit@4={result['hit_at_4']:.1f} | "
            f"Behavior={result['answer_behavior_correct']} | "
            f"Latency={result['latency_seconds']:.2f}s"
        )


if __name__ == "__main__":
    main()
