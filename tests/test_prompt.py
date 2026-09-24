from rag.prompt import create_rag_prompt


def test_rag_prompt_has_context_variable():
    prompt = create_rag_prompt()
    assert "context" in prompt.input_variables


def test_rag_prompt_has_question_variable():
    prompt = create_rag_prompt()
    assert "question" in prompt.input_variables


def test_rag_prompt_requires_context():
    prompt = create_rag_prompt()
    assert "CONTEXT" in prompt.template


def test_rag_prompt_requires_question():
    prompt = create_rag_prompt()
    assert "QUESTION" in prompt.template


def test_rag_prompt_has_missing_information_rule():
    prompt = create_rag_prompt()

    assert (
        "I could not find the answer in the provided document."
        in prompt.template
    )


def test_rag_prompt_prevents_outside_knowledge():
    prompt = create_rag_prompt()
    assert "Do not use outside knowledge." in prompt.template


def test_rag_prompt_prevents_invention():
    prompt = create_rag_prompt()
    assert "Do not invent information." in prompt.template


def test_rag_prompt_has_expected_variables_only():
    prompt = create_rag_prompt()

    assert set(prompt.input_variables) == {
        "context",
        "question",
    }
