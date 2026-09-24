from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    """
    Split PDF documents into retrieval-ready chunks.

    The original document metadata, including page and source,
    is preserved automatically.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were created from the documents.")

    return chunks
