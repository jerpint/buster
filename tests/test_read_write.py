import pandas as pd

from buster.busterbot import BusterAnswer
from buster.completers.base import Completion


class MockValidator:
    def check_sources_used(self, completion: Completion) -> bool:
        return True


def test_read_write_completion():
    c = Completion(error=False, completor="This is my completed answer")

    c_json = c.to_json()
    c_back = Completion.from_dict(c_json)

    assert c.error == c_back.error
    assert c.text == c.text
    assert c.version == c_back.version


def test_read_write_busteranswer():
    n_samples = 3
    b = BusterAnswer(
        user_input="This is my input",
        completion=Completion(error=False, completor="This is my completed answer"),
        validator=MockValidator(),
        matched_documents=pd.DataFrame.from_dict(
            {
                "title": ["test"] * n_samples,
                "url": ["http://url.com"] * n_samples,
                "content": ["cool text"] * n_samples,
                "embedding": [[0.0] * 1000] * n_samples,
                "n_tokens": [10] * n_samples,
                "source": ["fake source"] * n_samples,
            }
        ),
    )

    b_json = b.to_json()
    b_back = BusterAnswer.from_dict(b_json)

    assert b.version == b_back.version
    assert b.user_input == b_back.user_input
    assert b.completion.error == b_back.completion.error
    assert b.completion.text == b_back.completion.text
    assert b.completion.version == b_back.completion.version
    assert b.documents_relevant == b_back.documents_relevant
    for col in b_back.matched_documents.columns.tolist():
        assert col in b.matched_documents.columns.tolist()
        assert b_back.matched_documents[col].tolist() == b.matched_documents[col].tolist()
