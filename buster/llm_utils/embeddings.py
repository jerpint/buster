import logging
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm.contrib.concurrent import thread_map

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_openai_embedding_constructor(client_kwargs: Optional[dict] = None, model: str = "text-embedding-ada-002"):
    if client_kwargs is None:
        client_kwargs = {}
    client = OpenAI(**client_kwargs)

    @lru_cache
    def embedding_fn(text: str, model: str = model) -> np.array:
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

    return embedding_fn


# default embedding function
get_openai_embedding = get_openai_embedding_constructor()


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
    embeddings = thread_map(embedding_fn, df.content.to_list(), max_workers=num_workers)

    logger.info(f"Finished computing embeddings")
    return embeddings
