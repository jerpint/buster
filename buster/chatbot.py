import logging

import numpy as np
import openai
import pandas as pd
from openai.embeddings_utils import cosine_similarity, get_embedding

from buster.docparser import EMBEDDING_MODEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# search through the reviews for a specific product
def rank_documents(df: pd.DataFrame, query: str, top_k: int = 3) -> pd.DataFrame:
    product_embedding = get_embedding(
        query,
        engine=EMBEDDING_MODEL,
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    if top_k == -1:
        # return all results
        n = len(df)

    results = df.sort_values("similarity", ascending=False).head(top_k)
    return results


def engineer_prompt(question: str, documents: list[str]) -> str:
    return " ".join(documents) + "\nNow answer the following question:\n" + question


def get_gpt_response(question: str, df) -> str:
    # rank the documents, get the highest scoring doc and generate the prompt
    candidates = rank_documents(df, query=question, top_k=1)
    documents = candidates.text.to_list()
    prompt = engineer_prompt(question, documents)

    logger.info(f"querying GPT...")
    logger.info(f"User Question:\n{question}")
    # Call the API to generate a response
    try:
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
        logger.info(
            f"""
        GPT Response:\n{response_text}
        """
        )
        return response_text
    except Exception as e:
        import traceback

        logging.error(traceback.format_exc())
        return "Oops, something went wrong. Try again later!"


def load_embeddings(path: str) -> pd.DataFrame:
    logger.info(f"loading embeddings from {path}...")
    df = pd.read_csv(path)
    df["embedding"] = df.embedding.apply(eval).apply(np.array)
    logger.info(f"embeddings loaded.")
    return df


if __name__ == "__main__":
    # we generate the embeddings using docparser.py
    df = load_embeddings("data/document_embeddings.csv")

    question = "Where should I put my datasets when I am running a job?"
    response = get_gpt_response(question, df)
