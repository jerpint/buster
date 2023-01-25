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


def format_response(response_text, sources_url=None):

    response = f"{response_text}\n"

    if sources_url:
        response += f"\nHere are the sources I used to answer your response: \n{sources_url}\n"

    response += """
    \n\n
    I'm a bot 🤖 and not always perfect.
    For more info, view the full documentation here (https://docs.mila.quebec/) or contact support@mila.quebec
    """
    return response


def answer_question(question: str, df) -> str:
    # rank the documents, get the highest scoring doc and generate the prompt
    candidates = rank_documents(df, query=question, top_k=1)
    documents = candidates.text.to_list()
    sources_url = candidates.url.to_list()
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
        return format_response(response_text, sources_url)

    except Exception as e:
        import traceback

        logging.error(traceback.format_exc())
        response = "Oops, something went wrong. Try again later!"
        return format_response(response)


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
    response = answer_question(question, df)
