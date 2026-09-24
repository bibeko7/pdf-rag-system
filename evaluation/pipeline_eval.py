from evaluation.llm_eval import is_refusal_answer
from evaluation.retriever_eval import hit_at_k


def evaluate_pipeline(rag_pipeline, dataset):
    """Evaluate the complete RAG pipeline."""

    results = []

    for item in dataset:
        question_id = item["id"]
        question = item["question"]
        question_type = item["type"]
        relevant_pages = item["relevant_pages"]

        result = rag_pipeline.ask(question)

        retrieved_pages = []

        for document in result.retrieved_documents:
            page = document.metadata.get("page")

            if page is not None:
                retrieved_pages.append(int(page) + 1)

        pipeline_hit_at_4 = hit_at_k(
            retrieved_pages=retrieved_pages,
            relevant_pages=relevant_pages,
            k=4,
        )

        refusal = is_refusal_answer(result.answer)

        if question_type == "answerable":
            answer_behavior_correct = not refusal
        elif question_type == "out_of_context":
            answer_behavior_correct = refusal
        else:
            answer_behavior_correct = False

        results.append(
            {
                "id": question_id,
                "question": question,
                "type": question_type,
                "answer": result.answer,
                "retrieved_pages": retrieved_pages,
                "source_pages": [
                    (source.page + 1) if source.page is not None else None
                    for source in result.sources
                ],
                "hit_at_4": pipeline_hit_at_4,
                "refusal": refusal,
                "answer_behavior_correct": answer_behavior_correct,
                "latency_seconds": result.latency_seconds,
            }
        )

    total = len(results)

    if total == 0:
        raise ValueError("Dataset is empty.")

    answer_behavior_accuracy = sum(
        result["answer_behavior_correct"]
        for result in results
    ) / total

    retrieval_hit_at_4 = sum(
        result["hit_at_4"]
        for result in results
    ) / total

    average_latency = sum(
        result["latency_seconds"]
        for result in results
    ) / total

    return {
        "summary": {
            "total_questions": total,
            "answer_behavior_accuracy": answer_behavior_accuracy,
            "retrieval_hit_at_4": retrieval_hit_at_4,
            "average_latency_seconds": average_latency,
        },
        "results": results,
    }
