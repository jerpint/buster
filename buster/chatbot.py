import logging
import os
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import openai
import pandas as pd
import promptlayer
from openai.embeddings_utils import cosine_similarity, get_embedding

from buster.docparser import read_documents
from buster.formatter import Formatter, HTMLFormatter, MarkdownFormatter, SlackFormatter
from buster.formatter.base import Response, Source

FORMATTERS = {"text": Formatter, "slack": SlackFormatter, "html": HTMLFormatter, "markdown": MarkdownFormatter}


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Check if an API key exists for promptlayer, if it does, use it
promptlayer_api_key = os.environ.get("PROMPTLAYER_API_KEY")
if promptlayer_api_key:
    logger.info("Enabling prompt layer...")
    promptlayer.api_key = promptlayer_api_key

    # replace openai with the promptlayer wrapper
    openai = promptlayer.openai
    openai.api_key = os.environ.get("OPENAI_API_KEY")


@dataclass
class ChatbotConfig:
    """Configuration object for a chatbot.

    documents_csv: Path to the csv file containing the documents and their embeddings.
    embedding_model: OpenAI model to use to get embeddings.
    top_k: Max number of documents to retrieve, ordered by cosine similarity
    thresh: threshold for cosine similarity to be considered
    max_words: maximum number of words the retrieved documents can be. Will truncate otherwise.
    completion_kwargs: kwargs for the OpenAI.Completion() method
    separator: the separator to use, can be either "\n" or <p> depending on rendering.
    link_format: the type of format to render links with, e.g. slack or markdown
    unknown_prompt: Prompt to use to generate the "I don't know" embedding to compare to.
    text_before_prompt: Text to prompt GPT with before the user prompt, but after the documentation.
    text_after_response: Generic response to add the the chatbot's reply.
    """

    documents_file: str = "buster/data/document_embeddings.csv"
    embedding_model: str = "text-embedding-ada-002"
    top_k: int = 3
    thresh: float = 0.7
    max_words: int = 3000
    unknown_threshold: float = 0.9  # set to 0 to deactivate

    completion_kwargs: dict = field(
        default_factory=lambda: {
            "engine": "text-davinci-003",
            "max_tokens": 200,
            "temperature": None,
            "top_p": None,
            "frequency_penalty": 1,
            "presence_penalty": 1,
        }
    )
    separator: str = "\n"
    link_format: str = "slack"
    unknown_prompt: str = "I Don't know how to answer your question."
    text_before_documents: str = "You are a chatbot answering questions.\n"
    text_before_prompt: str = "Answer the following question:\n"
    text_after_response: str = "I'm a chatbot, bleep bloop."


class Chatbot:
    def __init__(self, cfg: ChatbotConfig):
        # TODO: right now, the cfg is being passed as an omegaconf, is this what we want?
        self.cfg = cfg
        self._init_documents()
        self._init_unk_embedding()

    def _init_documents(self):
        filepath = self.cfg.documents_file
        logger.info(f"loading embeddings from {filepath}...")
        self.documents = read_documents(filepath)
        logger.info(f"embeddings loaded.")

    def _init_unk_embedding(self):
        logger.info("Generating UNK embedding...")
        self.unk_embedding = get_embedding(
            self.cfg.unknown_prompt,
            engine=self.cfg.embedding_model,
        )

    def rank_documents(
        self,
        documents: pd.DataFrame,
        query: str,
        top_k: float,
        thresh: float,
        engine: str,
    ) -> pd.DataFrame:
        """
        Compare the question to the series of documents and return the best matching documents.
        """

        query_embedding = get_embedding(
            query,
            engine=engine,
        )
        documents["similarity"] = documents.embedding.apply(lambda x: cosine_similarity(x, query_embedding))

        # sort the matched_documents by score
        matched_documents = documents.sort_values("similarity", ascending=False)

        # limit search to top_k matched_documents.
        top_k = len(matched_documents) if top_k == -1 else top_k
        matched_documents = matched_documents.head(top_k)

        # log matched_documents to the console
        logger.info(f"matched documents before thresh: {matched_documents}")

        # filter out matched_documents using a threshold
        if thresh:
            matched_documents = matched_documents[matched_documents.similarity > thresh]
            logger.info(f"matched documents after thresh: {matched_documents}")

        return matched_documents

    def prepare_documents(self, matched_documents: pd.DataFrame, max_words: int) -> str:
        # gather the documents in one large plaintext variable
        documents_list = matched_documents.content.to_list()
        documents_str = " ".join(documents_list)

        # truncate the documents to fit
        # TODO: increase to actual token count
        word_count = len(documents_str.split(" "))
        if word_count > max_words:
            logger.info("truncating documents to fit...")
            documents_str = " ".join(documents_str.split(" ")[0:max_words])
            logger.info(f"Documents after truncation: {documents_str}")

        return documents_str

    def prepare_prompt(
        self,
        question: str,
        matched_documents: pd.DataFrame,
        text_before_prompt: str,
        text_before_documents: str,
    ) -> str:
        """
        Prepare the prompt with prompt engineering.
        """
        documents_str: str = self.prepare_documents(matched_documents, max_words=self.cfg.max_words)
        return text_before_documents + documents_str + text_before_prompt + question

    def get_gpt_response(self, **completion_kwargs) -> Response:
        # Call the API to generate a response
        logger.info(f"querying GPT...")
        try:
            response = openai.Completion.create(**completion_kwargs)
        except Exception as e:
            # log the error and return a generic response instead.
            logger.exception("Error connecting to OpenAI API. See traceback:")
            return Response("", True, "We're having trouble connecting to OpenAI right now... Try again soon!")

        text = response["choices"][0]["text"]
        return Response(text)

    def generate_response(
        self, prompt: str, matched_documents: pd.DataFrame, unknown_prompt: str
    ) -> tuple[Response, Iterable[Source]]:
        """
        Generate a response based on the retrieved documents.
        """
        if len(matched_documents) == 0:
            # No matching documents were retrieved, return
            sources = tuple()
            return Response(unknown_prompt), sources

        logger.info(f"Prompt:  {prompt}")
        response = self.get_gpt_response(prompt=prompt, **self.cfg.completion_kwargs)
        if response:
            logger.info(f"GPT Response:\n{response.text}")
            relevant = self.check_response_relevance(
                response=response.text,
                engine=self.cfg.embedding_model,
                unk_embedding=self.unk_embedding,
                unk_threshold=self.cfg.unknown_threshold,
            )
            if relevant:
                sources = (
                    Source(dct["name"], dct["url"], dct["similarity"])
                    for dct in matched_documents.to_dict(orient="records")
                )
            else:
                sources = tuple()

        return response, sources

    def check_response_relevance(
        self, response: str, engine: str, unk_embedding: np.array, unk_threshold: float
    ) -> bool:
        """Check to see if a response is relevant to the chatbot's knowledge or not.

        We assume we've prompt-engineered our bot to say a response is unrelated to the context if it isn't relevant.
        Here, we compare the embedding of the response to the embedding of the prompt-engineered "I don't know" embedding.

        set the unk_threshold to 0 to essentially turn off this feature.
        """
        response_embedding = get_embedding(
            response,
            engine=engine,
        )
        score = cosine_similarity(response_embedding, unk_embedding)
        logger.info(f"UNK score: {score}")

        # Likely that the answer is meaningful, add the top sources
        return score < unk_threshold

    def process_input(self, question: str, formatter: Formatter = None) -> str:
        """
        Main function to process the input question and generate a formatted output.
        """

        if formatter is None and self.cfg.link_format not in FORMATTERS:
            raise ValueError(f"Unknown link format {self.cfg.link_format}")
        elif formatter is None:
            formatter = FORMATTERS[self.cfg.link_format]()

        logger.info(f"User Question:\n{question}")

        matched_documents = self.rank_documents(
            documents=self.documents,
            query=question,
            top_k=self.cfg.top_k,
            thresh=self.cfg.thresh,
            engine=self.cfg.embedding_model,
        )
        prompt = self.prepare_prompt(
            question=question,
            matched_documents=matched_documents,
            text_before_prompt=self.cfg.text_before_prompt,
            text_before_documents=self.cfg.text_before_documents,
        )
        response, sources = self.generate_response(prompt, matched_documents, self.cfg.unknown_prompt)

        return formatter(response, sources)
