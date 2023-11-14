import pandas as pd

from buster.llm_utils import get_openai_embedding
from buster.validators import Validator

validator_cfg = {
    "use_reranking": True,
    "validate_documents": True,
    "answer_validator_cfg": {
        "unknown_response_templates": [
            "I Don't know how to answer your question.",
        ],
        "unknown_threshold": 0.85,
    },
    "question_validator_cfg": {
        "invalid_question_response": "This question does not seem relevant to my current knowledge.",
        "completion_kwargs": {
            "model": "gpt-3.5-turbo",
            "stream": False,
            "temperature": 0,
        },
        "check_question_prompt": "You are validating if questions are related to AI. If a question is relevant, respond with 'true', if it is irrlevant, respond with 'false'.",
    },
}
validator = Validator(**validator_cfg)


def test_validator_check_question_relevance():
    question = "What is backpropagation?"
    relevance, _ = validator.check_question_relevance(question)
    assert relevance == True

    question = "How can I make a broccoli soup?"
    relevance, _ = validator.check_question_relevance(question)
    assert relevance == False


def test_validator_check_answer_relevance():
    answer = "Not sure how to answer your question"
    assert validator.check_answer_relevance(answer) == False

    answer = "According to the documentation, the answer should be 2+2 = 4."
    assert validator.check_answer_relevance(answer) == True


def test_validator_check_documents_relevance():
    docs = {
        "content": [
            "A panda is a bear native to China, known for its black and white fur.",
            "An apple is a sweet fruit, often red, green, or yellow in color.",
            "A car is a wheeled vehicle used for transportation, typically powered by an engine.",
        ]
    }

    answer = "Pandas live in China."
    expected_relevance = [True, False, False]

    matched_documents = pd.DataFrame(docs)
    matched_documents = validator.check_documents_relevance(answer=answer, matched_documents=matched_documents)

    assert "relevance" in matched_documents.columns
    assert matched_documents.relevance.to_list() == expected_relevance


def test_validator_rerank_docs():
    documents = [
        "A basketball player practicing",
        "A cat eating an orange",
        "A green apple on the counter",
    ]
    matched_documents = pd.DataFrame({"documents": documents})
    matched_documents["embedding"] = matched_documents.documents.apply(lambda x: get_openai_embedding(x))

    answer = "An apple is a delicious fruit."
    reranked_documents = validator.rerank_docs(answer, matched_documents)

    assert reranked_documents.documents.to_list() == [
        "A green apple on the counter",
        "A cat eating an orange",
        "A basketball player practicing",
    ]
