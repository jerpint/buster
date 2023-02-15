import numpy as np
import pandas as pd

from buster.db import DocumentsDB


def test_read_write():
    db = DocumentsDB(":memory:")

    data = pd.DataFrame.from_dict(
        {
            "title": ["test"],
            "url": ["http://url.com"],
            "content": ["cool text"],
            "embedding": [np.arange(10, dtype=np.float32) - 0.3],
            "n_tokens": [10],
        }
    )
    db.reset_document_source("test", data)

    db_data = db.get_documents("test")

    assert db_data["title"].iloc[0] == data["title"].iloc[0]
    assert db_data["url"].iloc[0] == data["url"].iloc[0]
    assert db_data["content"].iloc[0] == data["content"].iloc[0]
    assert np.allclose(db_data["embedding"].iloc[0], data["embedding"].iloc[0])
    assert db_data["n_tokens"].iloc[0] == data["n_tokens"].iloc[0]
