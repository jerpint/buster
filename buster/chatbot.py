import logging
from dataclasses import dataclass, field

import numpy as np
import openai
import pandas as pd
from openai.embeddings_utils import cosine_similarity, get_embedding

from buster.docparser import EMBEDDING_MODEL, read_documents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class ChatbotConfig:
    """Configuration object for a chatbot.

    documents_csv: Path to the csv file containing the documents and their embeddings.
    embedding_model: OpenAI model to use to get embeddings.
    top_k: Max number of documents to retrieve, ordered by cosine similarity
    thresh: threshold for cosine similarity to be considered
    max_chars: maximum number of characters the retrieved documents can be. Will truncate otherwise.
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
    max_chars: int = 3000

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
    text_before_prompt: str = "I'm a chatbot, bleep bloop."
    text_after_response: str = "Answer the following question:\n"


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
        logger.info("Generating UNK token...")
        unknown_prompt = self.cfg.unknown_prompt
        engine = self.cfg.embedding_model
        self.unk_embedding = get_embedding(
            unknown_prompt,
            engine=engine,
        )

    def rank_documents(
        self,
        documents: pd.DataFrame,
        query: str,
    ) -> pd.DataFrame:
        """
        Compare the question to the series of documents and return the best matching documents.
        """
        top_k = self.cfg.top_k
        thresh = self.cfg.thresh
        engine = self.cfg.embedding_model  # EMBEDDING_MODEL

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

    def prepare_prompt(self, question: str, candidates: pd.DataFrame) -> str:
        """
        Prepare the prompt with prompt engineering.
        """

        max_chars = self.cfg.max_chars
        text_before_prompt = self.cfg.text_before_prompt

        documents_list = candidates.text.to_list()
        documents_str = " ".join(documents_list)
        if len(documents_str) > max_chars:
            logger.info("truncating documents to fit...")
            documents_str = documents_str[0:max_chars]

        return documents_str + text_before_prompt + question

    def generate_response(self, prompt: str, matched_documents: pd.DataFrame) -> str:
        """
        Generate a response based on the retrieved documents.
        """
        if len(matched_documents) == 0:
            # No matching documents were retrieved, return
            response_text = "I did not find any relevant documentation related to your question."
            return response_text

        logger.info(f"querying GPT...")
        logger.info(f"Prompt:  {prompt}")
        # Call the API to generate a response
        try:
            completion_kwargs = self.cfg.completion_kwargs
            completion_kwargs["prompt"] = prompt
            response = openai.Completion.create(**completion_kwargs)

            # Get the response text
            response_text = response["choices"][0]["text"]
            logger.info(f"GPT Response:\n{response_text}")
            return response_text

        except Exception as e:
            # log the error and return a generic response instead.
            import traceback

            logger.error("Error connecting to OpenAI API")
            logging.error(traceback.format_exc())
            response_text = "Hmm, we're having trouble connecting to OpenAI right now... Try again soon!"
            return response_text

    def add_sources(self, response: str, matched_documents: pd.DataFrame):
        """
        Add sources fromt the matched documents to the response.
        """
        sep = self.cfg.separator  # \n
        format = self.cfg.link_format

        urls = matched_documents.url.to_list()
        names = matched_documents.name.to_list()
        similarities = matched_documents.similarity.to_list()

        response += f"{sep}{sep}Here are the sources I used to answer your question:{sep}"
        for url, name, similarity in zip(urls, names, similarities):
            if format == "markdown":
                response += f"[{name}]({url}), relevance: {similarity:2.3f}{sep}"
            elif format == "slack":
                response += f"• <{url}|{name}>, relevance: {similarity:2.3f}{sep}"
            else:
                raise ValueError(f"{format} is not a valid URL format.")

        return response

    def format_response(self, response: str, matched_documents: pd.DataFrame) -> str:
        """
        Format the response by adding the sources if necessary, and a disclaimer prompt.
        """

        sep = self.cfg.separator
        text_after_response = self.cfg.text_after_response

        if len(matched_documents) > 0:
            # we have matched documents, now we check to see if the answer is meaningful
            response_embedding = get_embedding(
                response,
                engine=EMBEDDING_MODEL,
            )
            score = cosine_similarity(response_embedding, self.unk_embedding)
            logger.info(f"UNK score: {score}")
            if score < 0.9:
                # Liekly that the answer is meaningful, add the top sources
                response = self.add_sources(response, matched_documents=matched_documents)

        response += f"{sep}{sep}{sep}{text_after_response}{sep}"

        return response

    def process_input(self, question: str) -> str:
        """
        Main function to process the input question and generate a formatted output.
        """

        logger.info(f"User Question:\n{question}")

        matched_documents = self.rank_documents(documents=self.documents, query=question)
        prompt = self.prepare_prompt(question, matched_documents)
        response = self.generate_response(prompt, matched_documents)
        formatted_output = self.format_response(response, matched_documents)

        return formatted_output

