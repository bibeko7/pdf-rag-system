from rag.models import RAGResult, SourceInfo


def test_source_info():

    source = SourceInfo(
        source="test.pdf",
        page=5
    )

    assert source.source == "test.pdf"
    assert source.page == 5


def test_rag_result():

    result = RAGResult(
        question="What is machine learning?",
        answer="Machine learning learns patterns from data.",
        sources=[
            SourceInfo(
                source="test.pdf",
                page=2
            )
        ],
        latency_seconds=1.5
    )

    assert result.question == "What is machine learning?"
    assert result.answer != ""
    assert len(result.sources) == 1
    assert result.sources[0].page == 2
    assert result.latency_seconds == 1.5
