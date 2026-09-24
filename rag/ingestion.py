from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path: str):
    """
    Load a PDF file and return a list of LangChain Documents.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of LangChain Document objects.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the supplied file is not a PDF.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    if not documents:
        raise ValueError("The PDF contains no readable pages.")

    return documents
