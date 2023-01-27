import logging

import numpy as np
import openai
import pandas as pd
from omegaconf import OmegaConf
from openai.embeddings_utils import cosine_similarity, get_embedding

from buster.docparser import EMBEDDING_MODEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_documents(path: str) -> pd.DataFrame:
    logger.info(f"loading embeddings from {path}...")
    df = pd.read_csv(path)
    df["embedding"] = df.embedding.apply(eval).apply(np.array)
    logger.info(f"embeddings loaded.")
    return df


class Chatbot:
    def __init__(self, cfg: OmegaConf):
        # TODO: right now, the cfg is being passed as an omegaconf, is this what we want?
        self.cfg = cfg
        self.documents = load_documents(self.cfg.documents_csv)
        self.init_unk_embedding()

    def init_unk_embedding(self):
        unk_template = self.cfg.unk_template
        engine = self.cfg.embedding_model
        self.unk_embedding = get_embedding(
            unk_template,
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
        prompt_before = self.cfg.prompt_before

        documents_list = candidates.text.to_list()
        documents_str = " ".join(documents_list)
        if len(documents_str) > max_chars:
            logger.info("truncating documents to fit...")
            documents_str = documents_str[0:max_chars]

        return documents_str + prompt_before + question

    def generate_response(self, prompt: str, matched_documents: pd.DataFrame) -> str:
        """
        Generate a response based on the retrieved documents.
        """
        if len(matched_documents) == 0:
            # No matching documents were retrieved, return
            response_text = "I did not find any relevant documentation related to your question."
            return response_text

        engine = self.cfg.completion_engine  # text-davinci-003
        max_tokens = self.cfg.max_tokens  # 200
        temperature = self.cfg.temperature  # None
        top_p = self.cfg.top_p  # None

        logger.info(f"querying GPT...")
        # Call the API to generate a response
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=1,
                presence_penalty=1,
            )

            # Get the response text
            response_text = response["choices"][0]["text"]
            logger.info(f"GPT Response:\n{response_text}")
            return response_text

        except Exception as e:
            # log the error and return a generic response instead.
            import traceback

            logging.error(traceback.format_exc())
            response_text = "Oops, something went wrong. Try again later!"
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

        response += f"{sep}{sep}Here are the sources I used to answer your question:\n"
        for url, name, similarity in zip(urls, names, similarities):
            if format == "html":
                response += f"{sep}[{name}]({url}){sep}"
            elif format == "slack":
                response += f"• <{url}|{name}>, score: {similarity:2.3f}{sep}"

        return response

    def format_response(self, response: str, matched_documents: pd.DataFrame) -> str:
        """
        Format the response by adding the sources if necessary, and a disclaimer prompt.
        """

        sep = self.cfg.separator
        prompt_after = self.cfg.prompt_after

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

        response += f"{sep}{sep}{sep}{prompt_after}{sep}"

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
