def _normalize_pages(pages):
    return {int(page) for page in pages}


def hit_at_k(retrieved_pages, relevant_pages, k):
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    relevant = _normalize_pages(relevant_pages)
    retrieved = list(retrieved_pages)[:k]

    if not relevant:
        return 0.0

    return float(any(page in relevant for page in retrieved))


def precision_at_k(retrieved_pages, relevant_pages, k):
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    relevant = _normalize_pages(relevant_pages)
    retrieved = list(retrieved_pages)[:k]

    if not retrieved:
        return 0.0

    relevant_retrieved = sum(
        1 for page in retrieved
        if page in relevant
    )

    return relevant_retrieved / len(retrieved)


def recall_at_k(retrieved_pages, relevant_pages, k):
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    relevant = _normalize_pages(relevant_pages)
    retrieved = list(retrieved_pages)[:k]

    if not relevant:
        return 0.0

    relevant_retrieved = sum(
        1 for page in relevant
        if page in retrieved
    )

    return relevant_retrieved / len(relevant)


def reciprocal_rank(retrieved_pages, relevant_pages):
    relevant = _normalize_pages(relevant_pages)

    if not relevant:
        return 0.0

    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(results):
    if not results:
        return 0.0

    reciprocal_ranks = [
        reciprocal_rank(
            result["retrieved_pages"],
            result["relevant_pages"],
        )
        for result in results
    ]

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def evaluate_retrieval(
    retrieved_pages,
    relevant_pages,
    k=4,
):
    return {
        f"hit@{k}": hit_at_k(
            retrieved_pages,
            relevant_pages,
            k,
        ),
        f"precision@{k}": precision_at_k(
            retrieved_pages,
            relevant_pages,
            k,
        ),
        f"recall@{k}": recall_at_k(
            retrieved_pages,
            relevant_pages,
            k,
        ),
        "reciprocal_rank": reciprocal_rank(
            retrieved_pages,
            relevant_pages,
        ),
    }
