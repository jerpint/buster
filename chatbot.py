import pickle

import numpy as np
import openai
import pandas as pd
import tiktoken
from openai.embeddings_utils import cosine_similarity, get_embedding

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_ENCODING = "cl100k_base"  # this the encoding for text-embedding-ada-002


def load_documents(fname: str):
    df = pd.DataFrame()

    with open(fname, "rb") as fp:
        documents = pickle.load(fp)
    df["documents"] = documents
    return df


# search through the reviews for a specific product
def rank_documents(df, query, top_k=3):
    product_embedding = get_embedding(
        query,
        engine=EMBEDDING_MODEL,
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    if top_k == -1:
        n = len(df)

    results = df.sort_values("similarity", ascending=False).head(top_k)
    return results


def compute_n_tokens(df):
    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    df["n_tokens"] = df.documents.apply(lambda x: len(encoding.encode(x)))
    return df


def precompute_embeddings(df):
    df["embedding"] = df.documents.apply(lambda x: get_embedding(x, engine=EMBEDDING_MODEL))
    return df


def engineer_prompt(question: str, documents: list[str]):
    return " ".join(documents) + "\nNow answer the following question:\n" + question


def get_gpt_response(question: str, df, verbose=False):

    # rank the documents, get the highest scoring doc and generate the prompt
    candidates = rank_documents(df, query=question, top_k=1)
    documents = candidates.documents.to_list()
    prompt = engineer_prompt(question, documents)

    # Call the API to generate a response
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200,
        #  temperature=0,
        #  top_p=0,
        frequency_penalty=1,
        presence_penalty=1,
    )

    # Get the response text
    response_text = response["choices"][0]["text"]
    if verbose:
        print(f"User Question:\n{question}")
        print("")
        print(f"GPT Response:\n{response_text}")
    return response_text


def generate_embeddings(fname, output_csv):
    # Get all documents and precompute their embeddings
    df = load_documents(fname)
    df = compute_n_tokens(df)
    df = precompute_embeddings(df)
    df.to_csv(output_csv)
    return df


def load_embeddings(path):
    df = pd.read_csv(path)
    df["embedding"] = df.embedding.apply(eval).apply(np.array)
    return df


if __name__ == "__main__":
    # embedding model parameters

    # Only needs to be generated once
    #  df = generate_embeddings(fname="data/sections.pkl", output_csv="data/document_embeddings.csv")

    df = load_embeddings("data/document_embeddings.csv")

    question = "Where should I put my datasets when I am running a job?"
    response = get_gpt_response(question, df, verbose=True)
