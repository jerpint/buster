import logging

import numpy as np
import openai
import pandas as pd
from openai.embeddings_utils import cosine_similarity, get_embedding

from buster.docparser import EMBEDDING_MODEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

UNK_EMBEDDING = get_embedding(
    "This doesn't seem to be related to cluster usage. I am not sure how to answer.",
    engine=EMBEDDING_MODEL,
)
# search through the reviews for a specific product
def rank_documents(df: pd.DataFrame, query: str, top_k: int = 1, thresh: float = None) -> pd.DataFrame:
    product_embedding = get_embedding(
        query,
        engine=EMBEDDING_MODEL,
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))

    # sort by score
    results = df.sort_values("similarity", ascending=False)

    # get top_k
    top_k = len(df) if top_k == -1 else top_k
    results = results.head(top_k)

    # print results before thresholding
    logger.info(results)

    # filter out based on threshold
    if thresh:
        results = results[results.similarity > thresh]

    return results


def engineer_prompt(question: str, documents: list[str]) -> str:
    documents_str = " ".join(documents)
    if len(documents_str) > 3000:
        logger.info("truncating documents to fit...")
        documents_str = documents_str[0:3000]

    # Links should follow slack syntax.
    # As a reminder, links in slack are formatted like this: <http://www.example.com|This message *is* a link>
    prompt = """
You are a slack chatbot assistant answering technical questions about a cluster.
Make sure to format your answers in Markdown format, including code block and snippets.
Do not include any links to urls or hyperlinks in your answers.

If you do not know the answer to a question, or if it is completely irrelevant to cluster usage, simply reply with:

'This doesn't seem to be related to cluster usage.'

For example:

What is the meaning of life on the cluster?

This doesn't seem to be related to cluster usage.

Now answer the following question:
"""

    return documents_str + prompt + question


def add_sources(response: str, candidates: pd.DataFrame, style: str):
    # get the sources
    sep = "<br>" if style == "html" else "\n"
    sep2 = "```" if style == "html" else "\n"

    urls = candidates.url.to_list()
    names = candidates.name.to_list()
    similarities = candidates.similarity.to_list()

    response += f"{sep}{sep}Here are the sources I used to answer your question:\n"
    for url, name, similarity in zip(urls, names, similarities):
        if style == "html":
            response += f"{sep}[{name}]({url}){sep}"
        else:
            response += f"• <{url}|{name}>, score: {similarity:2.3f}{sep}"

    return response


def format_response(response_text: str, candidates: pd.DataFrame = None, style="html"):

    sep = "<br>" if style == "html" else "\n"
    sep2 = "```" if style == "html" else "\n"

    response = f"{response_text}"

    if candidates is not None:
        response_embedding = get_embedding(
            response,
            engine=EMBEDDING_MODEL,
        )
        score = cosine_similarity(response_embedding, UNK_EMBEDDING)
        logger.info(f"UNK score: {score}")
        if score < 0.9:
            # more likely that it knows an answer at this point
            response = add_sources(response, candidates=candidates, style=style)

    response += f"{sep}"

    response += f"""{sep}
I'm a bot 🤖 and not always perfect.
For more info, view the full documentation here (https://docs.mila.quebec/) or contact support@mila.quebec
{sep}
"""

    return response


def answer_question(question: str, df, top_k: int = 1, thresh: float = None, style="html") -> str:
    # rank the documents, get the highest scoring doc and generate the prompt
    candidates = rank_documents(df, query=question, top_k=top_k, thresh=thresh)

    logger.info(f"candidate responses: {candidates}")

    if len(candidates) == 0:
        return format_response(
            "I did not find any relevant documentation related to your question.", candidates=None, style=style
        )

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
        return format_response(response_text, candidates=candidates, style=style)

    except Exception as e:
        import traceback

        logging.error(traceback.format_exc())
        response = "Oops, something went wrong. Try again later!"
        return format_response(response, candidates=None, style=style)


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
