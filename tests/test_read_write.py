import pandas as pd

from buster.completers import Completion


class MockValidator:
    def __init__(self):
        self.use_reranking = True

    def check_answer_relevance(self, completion: Completion) -> bool:
        return True

    def rerank_docs(self, answer: str, matched_documents: pd.DataFrame) -> bool:
        return matched_documents


def test_read_write_completion():
    n_samples = 3
    matched_documents = pd.DataFrame.from_dict(
        {
            "title": ["test"] * n_samples,
            "url": ["http://url.com"] * n_samples,
            "content": ["cool text"] * n_samples,
            "embedding": [[0.0] * 1000] * n_samples,
            "n_tokens": [10] * n_samples,
            "source": ["fake source"] * n_samples,
        }
    )
    c = Completion(
        user_input="What is the meaning of life?",
        error=False,
        answer_generator="This is my actual answer",
        matched_documents=matched_documents,
        validator=MockValidator(),
    )

    c_json = c.to_json()
    c_back = Completion.from_dict(c_json)

    assert c.error == c_back.error
    assert c.answer_text == c_back.answer_text
    assert c.user_input == c_back.user_input
    assert c.answer_relevant == c_back.answer_relevant
    for col in c_back.matched_documents.columns.tolist():
        assert col in c.matched_documents.columns.tolist()
        assert c_back.matched_documents[col].tolist() == c.matched_documents[col].tolist()
