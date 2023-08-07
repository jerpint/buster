import numpy as np
import pandas as pd
import pytest

from buster.documents_manager import DeepLakeDocumentsManager
from buster.retriever import DeepLakeRetriever


@pytest.mark.parametrize(
    "documents_manager, retriever",
    [(DeepLakeDocumentsManager, DeepLakeRetriever)],
)
def test_write_read(tmp_path, documents_manager, retriever):
    retriever_cfg = {
        "top_k": 3,
        "thresh": 0.7,
        "max_tokens": 2000,
        "embedding_model": "text-embedding-ada-002",
    }
    dm_path = tmp_path / "tmp_dir_2"
    retriever_cfg["path"] = dm_path

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

    dm = DeepLakeDocumentsManager(vector_store_path=dm_path)

    dm.add(df=data)
    dm_data = retriever(**retriever_cfg).get_documents("sourceA")

    assert dm_data["title"].iloc[0] == data["title"].iloc[0]
    assert dm_data["url"].iloc[0] == data["url"].iloc[0]
    assert dm_data["content"].iloc[0] == data["content"].iloc[0]
    assert dm_data["source"].iloc[0] == data["source"].iloc[0]
    assert np.allclose(dm_data["embedding"].iloc[0], data["embedding"].iloc[0])


@pytest.mark.parametrize(
    "documents_manager, retriever",
    [
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
    db_path = tmp_path / "tmp_dir"
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


def test_generate_embeddings(tmp_path, monkeypatch):
    # Create fake data
    df = pd.DataFrame.from_dict(
        {"title": ["test"], "url": ["http://url.com"], "content": ["cool text"], "source": ["my_source"]}
    )

    # Patch the get_embedding function to return a fixed, fake embedding
    fake_embedding = [-0.005, 0.0018]
    monkeypatch.setattr(
        "buster.documents_manager.DeepLakeDocumentsManager._compute_embeddings",
        lambda self, df: df.content.apply(lambda y: fake_embedding),
    )

    # Generate embeddings, store in a file
    path = tmp_path / f"test_document_embeddings"
    dm = DeepLakeDocumentsManager(path)
    dm.add(df)

    # Read the embeddings from the file

    retriever_cfg = {
        "path": path,
        "top_k": 3,
        "thresh": 0.85,
        "max_tokens": 3000,
        "embedding_model": "text-embedding-ada-002",
    }
    read_df = DeepLakeRetriever(**retriever_cfg).get_documents("my_source")

    # Check all the values are correct across the files
    assert df["title"].iloc[0] == df["title"].iloc[0] == read_df["title"].iloc[0]
    assert df["url"].iloc[0] == df["url"].iloc[0] == read_df["url"].iloc[0]
    assert df["content"].iloc[0] == df["content"].iloc[0] == read_df["content"].iloc[0]
    assert np.allclose(fake_embedding, read_df["embedding"].iloc[0])
