import pandas as pd

from buster.completers.base import Completion
from buster.validators.base import Validator


def test_validator_check_response_relevant():
    validator_cfg = {
        "unknown_prompt": "I Don't know how to answer your question.",
        "unknown_threshold": 0.85,
        "embedding_model": "text-embedding-ada-002",
        "use_reranking": True,
    }
    validator = Validator(validator_cfg)

    completion = Completion(error=False, completor="Not sure how to answer your question")
    assert validator.check_response_relevant(completion) == False

    completion = Completion(error=False, completor="According to the documentation, the answer should be 2+2 = 4.")
    assert validator.check_response_relevant(completion) == True


def test_validator_check_response_relevant__error():
    validator_cfg = {
        "unknown_prompt": "I Don't know how to answer your question.",
        "unknown_threshold": 0.85,
        "embedding_model": "text-embedding-ada-002",
        "use_reranking": True,
    }
    validator = Validator(validator_cfg)

    completion = Completion(error=True, completor="Something went wrong...")
    assert validator.check_response_relevant(completion) == False


def test_validator_rerank_docs():
    validator_cfg = {
        "unknown_prompt": "I Don't know how to answer your question.",
        "unknown_threshold": 0.85,
        "embedding_model": "text-embedding-ada-002",
        "use_reranking": True,
    }
    validator = Validator(validator_cfg)
    from openai.embeddings_utils import get_embedding

    completion = Completion(completor="An apple is a delicious fruit.", error=False)
    matched_documents = pd.DataFrame(
        {
            "documents": [
                "A basketball player practicing",
                "A cat eating an orange",
                "A green apple on the counter",
            ]
        }
    )
    matched_documents["embedding"] = matched_documents.documents.apply(
        lambda x: get_embedding(x, engine=validator.embedding_model)
    )

    reranked_documents = validator.rerank_docs(completion, matched_documents)

    assert reranked_documents.documents.to_list() == [
        "A green apple on the counter",
        "A cat eating an orange",
        "A basketball player practicing",
    ]
