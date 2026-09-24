import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_ollama import ChatOllama


load_dotenv()


HUGGINGFACE_MODEL_NAME = "openai/gpt-oss-120b"
OLLAMA_MODEL_NAME = "gemma3:4b"

DEFAULT_LLM_PROVIDER = "huggingface"


def create_llm():
    provider = os.getenv(
        "LLM_PROVIDER",
        DEFAULT_LLM_PROVIDER,
    ).strip().lower()

    if provider == "ollama":
        return ChatOllama(
            model=OLLAMA_MODEL_NAME,
            temperature=0,
        )

    if provider == "huggingface":
        token = os.getenv(
            "HUGGINGFACEHUB_API_TOKEN"
        )

        if not token:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN is not configured."
            )

        llm = HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_MODEL_NAME,
            task="text-generation",
            huggingfacehub_api_token=token,
        )

        return ChatHuggingFace(llm=llm)

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {provider}. "
        "Use 'ollama' or 'huggingface'."
    )
