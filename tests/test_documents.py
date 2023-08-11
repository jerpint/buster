import os

import numpy as np
import pandas as pd
import pytest

from buster.documents_manager import DeepLakeDocumentsManager
from buster.documents_manager.base import (
    compute_embeddings_parallelized,
    get_embedding_openai,
)
from buster.retriever import DeepLakeRetriever

# Patch the get_embedding function to return a fixed, fake embedding
NUM_WORKERS = 1
fake_embedding = [-0.005, 0.0018]


def get_fake_embedding(*arg, **kwargs):
    return fake_embedding


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
    db.add(df=data_1, num_workers=NUM_WORKERS)

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
    db.add(df=data_2, num_workers=NUM_WORKERS)

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

    # Generate embeddings, store in a file
    path = tmp_path / f"test_document_embeddings"
    dm = DeepLakeDocumentsManager(path)
    dm.add(df, embedding_fn=get_fake_embedding, num_workers=NUM_WORKERS)

    # Read the embeddings from the file
    retriever_cfg = {
        "path": path,
        "top_k": 3,
        "thresh": 0.85,
        "max_tokens": 3000,
        "embedding_model": "fake-embedding",
    }
    read_df = DeepLakeRetriever(**retriever_cfg).get_documents("my_source")

    # Check all the values are correct across the files
    assert df["title"].iloc[0] == df["title"].iloc[0] == read_df["title"].iloc[0]
    assert df["url"].iloc[0] == df["url"].iloc[0] == read_df["url"].iloc[0]
    assert df["content"].iloc[0] == df["content"].iloc[0] == read_df["content"].iloc[0]
    assert np.allclose(fake_embedding, read_df["embedding"].iloc[0])


def test_generate_embeddings_parallelized():
    # Create fake data
    df = pd.DataFrame.from_dict(
        {
            "title": ["test"] * 5,
            "url": ["http://url.com"] * 5,
            "content": ["cool text" + str(x) for x in range(5)],
            "source": ["my_source"] * 5,
        }
    )

    embeddings_parallel = compute_embeddings_parallelized(
        df, embedding_fn=get_embedding_openai, num_workers=NUM_WORKERS
    )
    embeddings = df.content.apply(get_embedding_openai)

    # embeddings comes out as a series because of the apply, so cast it back to an array
    embeddings_arr = np.array(embeddings.to_list())

    assert np.allclose(embeddings_parallel, embeddings_arr, atol=1e-3)


def test_add_batches(tmp_path):
    dm_path = tmp_path / "deeplake_store"
    num_samples = 20
    batch_size = 16
    csv_filename = os.path.join(tmp_path, "embedding_")

    dm = DeepLakeDocumentsManager(vector_store_path=dm_path)

    # Create fake data
    df = pd.DataFrame.from_dict(
        {
            "title": ["test"] * num_samples,
            "url": ["http://url.com"] * num_samples,
            "content": ["cool text" + str(x) for x in range(num_samples)],
            "source": ["my_source"] * num_samples,
        }
    )

    dm.batch_add(
        df,
        embedding_fn=get_fake_embedding,
        num_workers=NUM_WORKERS,
        batch_size=batch_size,
        min_time_interval=0,
        csv_filename=csv_filename,
    )

    csv_files = [f for f in os.listdir(tmp_path) if f.endswith(".csv")]

    # check that we registered the good number of doucments and that files were generated
    assert len(dm) == num_samples

    df_saved = pd.read_csv(csv_filename)
    assert len(df_saved) == num_samples
    assert "embedding" in df_saved.columns
