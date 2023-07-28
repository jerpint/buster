import numpy as np
import pandas as pd
import pytest

from buster.documents import DocumentsDB, DeepLakeDocumentsManager
from buster.retriever import SQLiteRetriever, DeepLakeRetriever


@pytest.mark.parametrize(
    "documents_manager, retriever",
    [
        (DocumentsDB, SQLiteRetriever),
        (DeepLakeDocumentsManager, DeepLakeRetriever),
    ],
)
def test_write_read(tmp_path, documents_manager, retriever):
    retriever_cfg = {
        "top_k": 3,
        "thresh": 0.7,
        "max_tokens": 2000,
        "embedding_model": "text-embedding-ada-002",
    }
    if documents_manager is DocumentsDB:
        db_path = tmp_path / "test.db"
        retriever_cfg["db_path"] = db_path
    elif documents_manager is DeepLakeDocumentsManager:
        db_path = tmp_path / "deeplake"
        retriever_cfg["path"] = db_path

    db = documents_manager(db_path)
    data = pd.DataFrame.from_dict(
        {
            "title": ["test"],
            "url": ["http://url.com"],
            "content": ["cool text"],
            "source": ["sourceA"],
            "embedding": [np.arange(10, dtype=np.float32) - 0.3],
            "n_tokens": 5,
        }
    )
    db.add(df=data)
    db_data = retriever(**retriever_cfg).get_documents("sourceA")

    assert db_data["title"].iloc[0] == data["title"].iloc[0]
    assert db_data["url"].iloc[0] == data["url"].iloc[0]
    assert db_data["content"].iloc[0] == data["content"].iloc[0]
    assert db_data["source"].iloc[0] == data["source"].iloc[0]
    assert np.allclose(db_data["embedding"].iloc[0], data["embedding"].iloc[0])


@pytest.mark.parametrize(
    "documents_manager, retriever",
    [
        (DocumentsDB, SQLiteRetriever),
        (DeepLakeDocumentsManager, DeepLakeRetriever),
    ],
)
def test_write_write_read(tmp_path, documents_manager, retriever):
    retriever_cfg = {
        "top_k": 3,
        "thresh": 0.7,
        "max_tokens": 2000,
        "embedding_model": "text-embedding-ada-002",
    }
    if documents_manager is DocumentsDB:
        db_path = tmp_path / "test.db"
        retriever_cfg["db_path"] = db_path
    elif documents_manager is DeepLakeDocumentsManager:
        db_path = tmp_path / "deeplake"
        retriever_cfg["path"] = db_path

    db = documents_manager(db_path)

    data_1 = pd.DataFrame.from_dict(
        {
            "title": ["test"],
            "url": ["http://url.com"],
            "content": ["cool text"],
            "embedding": [np.arange(10, dtype=np.float32) - 0.3],
            "source": ["sourceA"],
            "n_tokens": 10,
        }
    )
    db.add(df=data_1)

    data_2 = pd.DataFrame.from_dict(
        {
            "title": ["other"],
            "url": ["http://url.com/page.html"],
            "content": ["lorem ipsum"],
            "embedding": [np.arange(10, dtype=np.float32) / 10 - 2.3],
            "source": ["sourceB"],
            "n_tokens": 5,
        }
    )
    db.add(df=data_2)

    db_data = retriever(**retriever_cfg).get_documents("sourceB")

    assert len(db_data) == len(data_2)
    assert db_data["title"].iloc[0] == data_2["title"].iloc[0]
    assert db_data["url"].iloc[0] == data_2["url"].iloc[0]
    assert db_data["content"].iloc[0] == data_2["content"].iloc[0]
    assert np.allclose(db_data["embedding"].iloc[0], data_2["embedding"].iloc[0])


def test_update_source(tmp_path):
    display_name = "Super Test"
    db_path = tmp_path / "test.db"
    db = DocumentsDB(db_path)

    db.update_source(source="sourceA", display_name=display_name)

    retriever_cfg = {
        "db_path": db_path,
        "top_k": 3,
        "thresh": 0.7,
        "max_tokens": 2000,
        "embedding_model": "text-embedding-ada-002",
    }

    returned_display_name = SQLiteRetriever(**retriever_cfg).get_source_display_name("sourceA")

    assert display_name == returned_display_name
