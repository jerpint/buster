import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from openai.embeddings_utils import cosine_similarity, get_embedding

from buster.completers.base import get_completer
from buster.documents import get_documents_manager_from_extension
from buster.formatter import (
    Response,
    ResponseFormatter,
    Source,
    response_formatter_factory,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class BusterConfig:
    """Configuration object for a chatbot.

    documents_csv: Path to the csv file containing the documents and their embeddings.
    embedding_model: OpenAI model to use to get embeddings.
    top_k: Max number of documents to retrieve, ordered by cosine similarity
    thresh: threshold for cosine similarity to be considered
    max_words: maximum number of words the retrieved documents can be. Will truncate otherwise.
    completion_kwargs: kwargs for the OpenAI.Completion() method
    separator: the separator to use, can be either "\n" or <p> depending on rendering.
    response_format: the type of format to render links with, e.g. slack or markdown
    unknown_prompt: Prompt to use to generate the "I don't know" embedding to compare to.
    text_before_prompt: Text to prompt GPT with before the user prompt, but after the documentation.
    reponse_footnote: Generic response to add the the chatbot's reply.
    """

    documents_file: str = "buster/data/document_embeddings.tar.gz"
    embedding_model: str = "text-embedding-ada-002"
    top_k: int = 3
    thresh: float = 0.7
    max_words: int = 3000
    unknown_threshold: float = 0.9  # set to 0 to deactivate
    completer_cfg: dict = field(
        # TODO: Put all this in its own config with sane defaults?
        default_factory=lambda: {
            "name": "GPT3",
            "text_before_documents": "You are a chatbot answering questions.\n",
            "text_before_prompt": "Answer the following question:\n",
            "completion_kwargs": {
                "engine": "text-davinci-003",
                "max_tokens": 200,
                "temperature": None,
                "top_p": None,
                "frequency_penalty": 1,
                "presence_penalty": 1,
            },
        }
    )
    separator: str = "\n"
    response_format: str = "slack"
    unknown_prompt: str = "I Don't know how to answer your question."
    response_footnote: str = "I'm a bot 🤖 and not always perfect."


class Buster:
    def __init__(self, cfg: BusterConfig):
        # TODO: right now, the cfg is being passed as an omegaconf, is this what we want?
        self.cfg = cfg
        self.completer = get_completer(cfg.completer_cfg)
        self._init_documents()
        self._init_unk_embedding()
        self._init_response_formatter()

    def _init_response_formatter(self):
        self.response_formatter = response_formatter_factory(
            format=self.cfg.response_format, response_footnote=self.cfg.response_footnote
        )

    def _init_documents(self):
        filepath = self.cfg.documents_file
        logger.info(f"loading embeddings from {filepath}...")
        self.documents = get_documents_manager_from_extension(filepath)(filepath)
        logger.info(f"embeddings loaded.")

    def _init_unk_embedding(self):
        logger.info("Generating UNK embedding...")
        self.unk_embedding = get_embedding(
            self.cfg.unknown_prompt,
            engine=self.cfg.embedding_model,
        )

    def rank_documents(
        self,
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
        matched_documents = self.documents.retrieve(query_embedding, top_k)

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

    def add_sources(
        self,
        response,
        matched_documents: pd.DataFrame,
        unknown_prompt: str,
    ):
        logger.info(f"GPT Response:\n{response.text}")
        sources = (
            Source(dct["source"], dct["url"], dct["similarity"]) for dct in matched_documents.to_dict(orient="records")
        )

        return sources

    def check_response_relevance(
        self, completion: str, engine: str, unk_embedding: np.array, unk_threshold: float
    ) -> bool:
        """Check to see if a response is relevant to the chatbot's knowledge or not.

        We assume we've prompt-engineered our bot to say a response is unrelated to the context if it isn't relevant.
        Here, we compare the embedding of the response to the embedding of the prompt-engineered "I don't know" embedding.

        set the unk_threshold to 0 to essentially turn off this feature.
        """
        response_embedding = get_embedding(
            completion,
            engine=engine,
        )
        score = cosine_similarity(response_embedding, unk_embedding)
        logger.info(f"UNK score: {score}")

        # Likely that the answer is meaningful, add the top sources
        return score < unk_threshold

    def process_input(self, user_input: str, formatter: ResponseFormatter = None) -> str:
        """
        Main function to process the input question and generate a formatted output.
        """

        logger.info(f"User Input:\n{user_input}")

        # We make sure there is always a newline at the end of the question to avoid completing the question.
        if not user_input.endswith("\n"):
            user_input += "\n"

        matched_documents = self.rank_documents(
            query=user_input,
            top_k=self.cfg.top_k,
            thresh=self.cfg.thresh,
            engine=self.cfg.embedding_model,
        )

        if len(matched_documents) == 0:
            response = Response("I did not find any sources to answer your question.")
            sources = tuple()
            return self.response_formatter(response, sources)

        # generate a completion
        documents: str = self.prepare_documents(matched_documents, max_words=self.cfg.max_words)
        response = self.completer.generate_response(user_input, documents)
        sources = self.add_sources(response, matched_documents, self.cfg.unknown_prompt)

        # check for relevance
        relevant = self.check_response_relevance(
            completion=response.text,
            engine=self.cfg.embedding_model,
            unk_embedding=self.unk_embedding,
            unk_threshold=self.cfg.unknown_threshold,
        )
        if not relevant:
            # answer generated was the chatbot saying it doesn't know how to answer
            # override completion with generic "I don't know"
            response = Response(text=self.cfg.unknown_prompt)
            sources = tuple()

        return self.response_formatter(response, sources)
