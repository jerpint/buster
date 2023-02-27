import numpy as np
import pandas as pd
import pytest

from buster.documents import DocumentsDB, DocumentsPickle


@pytest.mark.parametrize("documents_manager, extension", [(DocumentsDB, "db"), (DocumentsPickle, "tar.gz")])
def test_write_read(tmp_path, documents_manager, extension):
    db = documents_manager(tmp_path / f"test.{extension}")

    data = pd.DataFrame.from_dict(
        {
            "title": ["test"],
            "url": ["http://url.com"],
            "content": ["cool text"],
            "embedding": [np.arange(10, dtype=np.float32) - 0.3],
            "n_tokens": [10],
        }
    )
    db.add(source="test", df=data)

    db_data = db.get_documents("test")

    assert db_data["title"].iloc[0] == data["title"].iloc[0]
    assert db_data["url"].iloc[0] == data["url"].iloc[0]
    assert db_data["content"].iloc[0] == data["content"].iloc[0]
    assert np.allclose(db_data["embedding"].iloc[0], data["embedding"].iloc[0])
    assert db_data["n_tokens"].iloc[0] == data["n_tokens"].iloc[0]


@pytest.mark.parametrize("documents_manager, extension", [(DocumentsDB, "db"), (DocumentsPickle, "tar.gz")])
def test_write_write_read(tmp_path, documents_manager, extension):
    db = documents_manager(tmp_path / f"test.{extension}")

    data_1 = pd.DataFrame.from_dict(
        {
            "title": ["test"],
            "url": ["http://url.com"],
            "content": ["cool text"],
            "embedding": [np.arange(10, dtype=np.float32) - 0.3],
            "n_tokens": [10],
        }
    )
    db.add(source="test", df=data_1)

    data_2 = pd.DataFrame.from_dict(
        {
            "title": ["other"],
            "url": ["http://url.com/page.html"],
            "content": ["lorem ipsum"],
            "embedding": [np.arange(20, dtype=np.float32) / 10 - 2.3],
            "n_tokens": [20],
        }
    )
    db.add(source="test", df=data_2)

    db_data = db.get_documents("test")

    assert len(db_data) == len(data_2)
    assert db_data["title"].iloc[0] == data_2["title"].iloc[0]
    assert db_data["url"].iloc[0] == data_2["url"].iloc[0]
    assert db_data["content"].iloc[0] == data_2["content"].iloc[0]
    assert np.allclose(db_data["embedding"].iloc[0], data_2["embedding"].iloc[0])
    assert db_data["n_tokens"].iloc[0] == data_2["n_tokens"].iloc[0]
