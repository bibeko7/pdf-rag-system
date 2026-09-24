from langchain_core.prompts import PromptTemplate


def create_rag_prompt():
    return PromptTemplate(
        template="""
You answer questions using the CONTEXT below.

IMPORTANT:
- If the answer is in the CONTEXT, answer the question.
- If the answer is NOT in the CONTEXT, say exactly:
  "I could not find the answer in the provided document."
- Use only the CONTEXT.
- Do not use outside knowledge.
- Do not invent information.
- Keep the answer concise.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
""",
        input_variables=["context", "question"],
    )
