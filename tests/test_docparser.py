import numpy as np
import pandas as pd

from buster.docparser import generate_embeddings, read_documents, write_documents


def test_generate_embeddings(tmp_path, monkeypatch):
    # Patch the get_embedding function to return a fixed embedding
    monkeypatch.setattr("buster.docparser.get_embedding", lambda x, engine: [-0.005, 0.0018])

    # Create fake data
    data = pd.DataFrame.from_dict({"title": ["test"], "url": ["http://url.com"], "content": ["cool text"]})

    # Write the data to a file
    filepath = tmp_path / "test_document.csv"
    write_documents(filepath=filepath, documents_df=data, source="test")

    # Generate embeddings, store in a file
    output_file = tmp_path / "test_document_embeddings.tar.gz"
    df = generate_embeddings(filepath=filepath, output_file=output_file, source="test")

    # Read the embeddings from the file
    read_df = read_documents(output_file, "test")

    # Check all the values are correct across the files
    assert df["title"].iloc[0] == data["title"].iloc[0] == read_df["title"].iloc[0]
    assert df["url"].iloc[0] == data["url"].iloc[0] == read_df["url"].iloc[0]
    assert df["content"].iloc[0] == data["content"].iloc[0] == read_df["content"].iloc[0]
    assert np.allclose(df["embedding"].iloc[0], read_df["embedding"].iloc[0])
    assert df["n_tokens"].iloc[0] == read_df["n_tokens"].iloc[0]
