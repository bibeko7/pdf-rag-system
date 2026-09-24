import time
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser

from .chunking import split_documents
from .document_manager import (
    create_collection_name,
    create_document_id,
    get_document_name,
)
from .embeddings import create_embeddings
from .ingestion import load_pdf
from .llm import create_llm
from .models import RAGResult, SourceInfo
from .prompt import create_rag_prompt
from .retriever import create_retriever
from .vectorstore import create_vectorstore


class RAGPipeline:
    """
    Complete PDF-based Retrieval-Augmented Generation pipeline.

    Workflow:

        ingest()
            PDF
             ?
            documents
             ?
            chunks
             ?
            embeddings
             ?
            ChromaDB
             ?
            retriever

        ask()
            question
             ?
            retrieval
             ?
            context
             ?
            prompt
             ?
            LLM
             ?
            RAGResult
    """

    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None

        self.llm = None
        self.prompt = None
        self.generation_chain = None

        self.pdf_path = None
        self.document_id = None
        self.document_name = None
        self.collection_name = None
        self.chunks = []

    def ingest(self, pdf_path: str) -> int:
        """
        Load, chunk, embed, and index a PDF.

        Returns:
            Number of indexed chunks.
        """

        path = Path(pdf_path)

        if not path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF files are supported."
            )

        self.pdf_path = str(path)

        self.document_name = get_document_name(
            self.pdf_path
        )

        self.document_id = create_document_id(
            self.pdf_path
        )

        self.collection_name = create_collection_name(
            self.document_id
        )

        documents = load_pdf(
            self.pdf_path
        )

        self.chunks = split_documents(
            documents
        )

        self.embeddings = create_embeddings()

        self.vectorstore = create_vectorstore(
            self.chunks,
            self.embeddings,
            self.collection_name
        )

        self.retriever = create_retriever(
            self.vectorstore
        )

        self.prompt = create_rag_prompt()

        return len(self.chunks)

    def _initialize_generation(self):
        """
        Initialize the LLM and generation chain lazily.

        The LLM is not initialized during PDF ingestion.
        """

        if self.generation_chain is not None:
            return

        self.llm = create_llm()

        self.generation_chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> RAGResult:
        """
        Ask a question about the indexed PDF.

        Retrieval is performed exactly once so that the
        retrieved documents used for generation are also
        the documents reported as sources.
        """

        if self.retriever is None:
            raise RuntimeError(
                "No PDF has been indexed. "
                "Call ingest() before ask()."
            )

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        question = question.strip()

        start_time = time.perf_counter()

        retrieved_documents = self.retriever.invoke(
            question
        )

        context = "\n\n".join(
            document.page_content
            for document in retrieved_documents
        )

        self._initialize_generation()

        answer = self.generation_chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        latency = time.perf_counter() - start_time

        sources = []
        seen_sources = set()

        for document in retrieved_documents:

            source = document.metadata.get(
                "source",
                self.document_name or "unknown"
            )

            page = document.metadata.get(
                "page"
            )

            source_key = (
                source,
                page
            )

            if source_key not in seen_sources:

                sources.append(
                    SourceInfo(
                        source=source,
                        page=page
                    )
                )

                seen_sources.add(
                    source_key
                )

        return RAGResult(
            question=question,
            answer=answer,
            sources=sources,
            retrieved_documents=retrieved_documents,
            latency_seconds=latency
        )
