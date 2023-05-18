import numpy as np
import pandas as pd
import pytest

from buster.docparser import generate_embeddings
from buster.utils import (
    get_documents_manager_from_extension,
    get_retriever_from_extension,
)


@pytest.mark.parametrize("extension", ["db", "tar.gz"])
def test_generate_embeddings(tmp_path, monkeypatch, extension):
    # Create fake data
    data = pd.DataFrame.from_dict(
        {"title": ["test"], "url": ["http://url.com"], "content": ["cool text"], "source": ["my_source"]}
    )

    # Patch the get_embedding function to return a fixed embedding
    monkeypatch.setattr("buster.docparser.get_embedding", lambda x, engine: [-0.005, 0.0018])
    monkeypatch.setattr("buster.docparser.get_all_documents", lambda a, b, c: data)

    # Generate embeddings, store in a file
    output_file = tmp_path / f"test_document_embeddings.{extension}"
    manager = get_documents_manager_from_extension(output_file)(output_file)
    df = generate_embeddings(data, manager)

    # Read the embeddings from the file

    retriever_cfg = {
        "db_path": output_file,
        "top_k": 3,
        "thresh": 0.85,
        "max_tokens": 3000,
        "embedding_model": "text-embedding-ada-002",
    }
    read_df = get_retriever_from_extension(output_file)(**retriever_cfg).get_documents("my_source")

    # Check all the values are correct across the files
    assert df["title"].iloc[0] == data["title"].iloc[0] == read_df["title"].iloc[0]
    assert df["url"].iloc[0] == data["url"].iloc[0] == read_df["url"].iloc[0]
    assert df["content"].iloc[0] == data["content"].iloc[0] == read_df["content"].iloc[0]
    assert np.allclose(df["embedding"].iloc[0], read_df["embedding"].iloc[0])
    assert df["n_tokens"].iloc[0] == read_df["n_tokens"].iloc[0]
