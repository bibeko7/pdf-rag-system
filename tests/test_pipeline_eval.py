from evaluation.pipeline_eval import evaluate_pipeline


class FakeDocument:
    def __init__(self, page):
        self.metadata = {"page": page}


class FakeSource:
    def __init__(self, page):
        self.page = page


class FakeResult:
    def __init__(self, answer, pages, latency=1.0):
        self.answer = answer
        self.retrieved_documents = [
            FakeDocument(page) for page in pages
        ]
        self.sources = [
            FakeSource(page) for page in pages
        ]
        self.latency_seconds = latency


class FakePipeline:
    def __init__(self, answers):
        self.answers = answers

    def ask(self, question):
        return self.answers[question]


def test_pipeline_evaluation_answerable():
    question = "What is ML?"

    dataset = [
        {
            "id": "q001",
            "question": question,
            "ground_truth": "Machine Learning...",
            "relevant_pages": [1],
            "type": "answerable",
        }
    ]

    pipeline = FakePipeline(
        {
            question: FakeResult(
                answer="Machine Learning is a branch of AI.",
                pages=[0],
            )
        }
    )

    result = evaluate_pipeline(pipeline, dataset)

    assert result["summary"]["total_questions"] == 1
    assert result["summary"]["answer_behavior_accuracy"] == 1.0
    assert result["summary"]["retrieval_hit_at_4"] == 1.0


def test_pipeline_evaluation_out_of_context():
    question = "What is quantum computing?"

    dataset = [
        {
            "id": "q015",
            "question": question,
            "ground_truth": "",
            "relevant_pages": [],
            "type": "out_of_context",
        }
    ]

    pipeline = FakePipeline(
        {
            question: FakeResult(
                answer="I could not find the answer in the provided document.",
                pages=[6, 0],
            )
        }
    )

    result = evaluate_pipeline(pipeline, dataset)

    assert result["summary"]["answer_behavior_accuracy"] == 1.0


def test_pipeline_evaluation_average_latency():
    dataset = [
        {
            "id": "q001",
            "question": "Q1",
            "ground_truth": "A1",
            "relevant_pages": [1],
            "type": "answerable",
        },
        {
            "id": "q002",
            "question": "Q2",
            "ground_truth": "A2",
            "relevant_pages": [2],
            "type": "answerable",
        },
    ]

    pipeline = FakePipeline(
        {
            "Q1": FakeResult("Answer 1", [0], 2.0),
            "Q2": FakeResult("Answer 2", [1], 4.0),
        }
    )

    result = evaluate_pipeline(pipeline, dataset)

    assert result["summary"]["average_latency_seconds"] == 3.0
