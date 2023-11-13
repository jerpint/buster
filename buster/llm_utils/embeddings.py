import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm.contrib.concurrent import process_map

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

client = OpenAI()


@lru_cache
def get_openai_embedding(text: str, model: str = "text-embedding-ada-002"):
    try:
        text = text.replace("\n", " ")
        response = client.embeddings.create(
            input=text,
            model=model,
        )
        embedding = response.data[0].embedding
        return np.array(embedding, dtype="float32")
    except Exception as e:
        # This rarely happens with the API but in the off chance it does, will allow us not to loose the progress.
        logger.exception(e)
        logger.warning(f"Embedding failed to compute for {text=}")
        return None


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_embeddings_parallelized(df: pd.DataFrame, embedding_fn: callable, num_workers: int) -> pd.Series:
    """Compute the embeddings on the 'content' column of a DataFrame in parallel.

    This method calculates embeddings for the entries in the 'content' column of the provided DataFrame using the specified
    embedding function. The 'content' column is expected to contain strings or textual data. The method processes the
    embeddings in parallel using the number of workers specified.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to compute embeddings for.
        embedding_fn (callable): A function that computes embeddings for a given input string.
        num_workers (int): The number of parallel workers to use for computing embeddings.

    Returns:
        pd.Series: A Series containing the computed embeddings for each entry in the 'content' column.
    """

    logger.info(f"Computing embeddings of {len(df)} chunks. Using {num_workers=}")
    embeddings = process_map(embedding_fn, df.content.to_list(), max_workers=num_workers)

    logger.info(f"Finished computing embeddings")
    return embeddings
