from rag.llm import (
    DEFAULT_LLM_PROVIDER,
    HUGGINGFACE_MODEL_NAME,
    OLLAMA_MODEL_NAME,
)


def test_default_llm_provider():
    assert DEFAULT_LLM_PROVIDER == "huggingface"


def test_huggingface_model_name():
    assert HUGGINGFACE_MODEL_NAME == "openai/gpt-oss-120b"


def test_ollama_model_name():
    assert OLLAMA_MODEL_NAME == "gemma3:4b"
