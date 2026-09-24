import json
import shutil
from pathlib import Path

from rag.chunking import split_documents
from rag.document_manager import create_collection_name, create_document_id
from rag.embeddings import create_embeddings
from rag.ingestion import load_pdf
from rag.retriever import create_retriever
from rag.vectorstore import create_vectorstore

from evaluation.dataset import load_evaluation_dataset
from evaluation.retriever_eval import evaluate_retrieval, mean_reciprocal_rank


PDF_PATH = "data/evaluation/Machine_Learning_No_Table.pdf"
DATASET_PATH = "data/evaluation/test_dataset.json"
RESULTS_PATH = "data/evaluation/retriever_results.json"
EVALUATION_CHROMA_DIRECTORY = "data/evaluation/chroma_db"

TOP_K = 4
FETCH_K = 12


def get_page_number(document):
    page = document.metadata.get("page")

    if page is None:
        return None

    # PyPDFLoader uses zero-based page numbers.
    # Evaluation dataset uses one-based human-readable pages.
    return int(page) + 1


def close_vectorstore(vectorstore):
    """Release Chroma resources before deleting/rebuilding its directory."""

    if vectorstore is None:
        return

    client = getattr(vectorstore, "_client", None)

    if client is not None:
        close_method = getattr(client, "close", None)

        if callable(close_method):
            try:
                close_method()
            except Exception:
                pass


def main():
    pdf_path = Path(PDF_PATH)
    dataset_path = Path(DATASET_PATH)
    results_path = Path(RESULTS_PATH)
    chroma_path = Path(EVALUATION_CHROMA_DIRECTORY)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Evaluation PDF not found: {pdf_path}"
        )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}"
        )

    dataset = load_evaluation_dataset(DATASET_PATH)

    print("=" * 70)
    print("RETRIEVER EVALUATION")
    print("=" * 70)
    print(f"PDF: {pdf_path}")
    print(f"Questions: {len(dataset)}")
    print(f"Top-K: {TOP_K}")
    print(f"Fetch-K: {FETCH_K}")
    print()

    print("Loading PDF...")
    documents = load_pdf(str(pdf_path))
    print(f"Pages loaded: {len(documents)}")

    print("Creating chunks...")
    chunks = split_documents(documents)
    print(f"Chunks created: {len(chunks)}")

    print("Loading embedding model...")
    embeddings = create_embeddings()

    document_id = create_document_id(str(pdf_path))
    collection_name = create_collection_name(
        f"eval_{document_id}"
    )

    # Remove the previous evaluation-only vector store.
    # Production chroma_db is never touched.
    if chroma_path.exists():
        print("Removing previous evaluation vector store...")

        try:
            shutil.rmtree(chroma_path)
        except PermissionError:
            raise RuntimeError(
                "The evaluation Chroma directory is locked. "
                "Stop any process using it and run the evaluation again."
            )

    chroma_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectorstore = None
    results = []

    try:
        print("Creating evaluation vector store...")
        print(f"Directory: {chroma_path}")

        vectorstore = create_vectorstore(
            chunks=chunks,
            embeddings=embeddings,
            collection_name=collection_name,
            persist_directory=str(chroma_path),
        )

        retriever = create_retriever(
            vectorstore,
            k=TOP_K,
            fetch_k=FETCH_K,
        )

        print()
        print("Running retrieval evaluation...")
        print()

        for item in dataset:
            question_id = item["id"]
            question = item["question"]
            relevant_pages = item["relevant_pages"]

            retrieved_documents = retriever.invoke(question)

            retrieved_pages = [
                page
                for page in (
                    get_page_number(document)
                    for document in retrieved_documents
                )
                if page is not None
            ]

            metrics = evaluate_retrieval(
                retrieved_pages=retrieved_pages,
                relevant_pages=relevant_pages,
                k=TOP_K,
            )

            result = {
                "id": question_id,
                "question": question,
                "type": item["type"],
                "relevant_pages": relevant_pages,
                "retrieved_pages": retrieved_pages,
                **metrics,
            }

            results.append(result)

            print(f"{question_id}: {question}")
            print(f"  Relevant pages : {relevant_pages}")
            print(f"  Retrieved pages: {retrieved_pages}")
            print(
                f"  Hit@{TOP_K}       : "
                f"{metrics[f'hit@{TOP_K}']:.2f}"
            )
            print(
                f"  Precision@{TOP_K}: "
                f"{metrics[f'precision@{TOP_K}']:.2f}"
            )
            print(
                f"  Recall@{TOP_K}   : "
                f"{metrics[f'recall@{TOP_K}']:.2f}"
            )
            print(
                f"  Reciprocal Rank : "
                f"{metrics['reciprocal_rank']:.2f}"
            )
            print()

    finally:
        close_vectorstore(vectorstore)

    mrr = mean_reciprocal_rank(results)

    hit_key = f"hit@{TOP_K}"
    precision_key = f"precision@{TOP_K}"
    recall_key = f"recall@{TOP_K}"

    mean_hit = sum(
        result[hit_key] for result in results
    ) / len(results)

    mean_precision = sum(
        result[precision_key] for result in results
    ) / len(results)

    mean_recall = sum(
        result[recall_key] for result in results
    ) / len(results)

    summary = {
        "questions_evaluated": len(results),
        "top_k": TOP_K,
        "fetch_k": FETCH_K,
        "hit_at_k": mean_hit,
        "precision_at_k": mean_precision,
        "recall_at_k": mean_recall,
        "mrr": mrr,
    }

    output = {
        "evaluation": "retriever",
        "pdf": str(pdf_path),
        "dataset": str(dataset_path),
        "summary": summary,
        "questions": results,
    }

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 70)
    print("OVERALL RETRIEVAL RESULTS")
    print("=" * 70)
    print(f"Questions evaluated : {len(results)}")
    print(f"Hit@{TOP_K}             : {mean_hit:.4f}")
    print(f"Precision@{TOP_K}      : {mean_precision:.4f}")
    print(f"Recall@{TOP_K}         : {mean_recall:.4f}")
    print(f"MRR                 : {mrr:.4f}")
    print()
    print(f"Results saved to: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
