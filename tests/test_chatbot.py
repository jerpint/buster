import numpy as np
import pandas as pd

from buster.chatbot import Chatbot, ChatbotConfig
from buster.documents import DocumentsManager


def get_fake_embedding(length=1536):
    rng = np.random.default_rng()
    return list(rng.random(length, dtype=np.float32))


class DocumentsMock(DocumentsManager):
    def __init__(self, filepath):
        self.filepath = filepath

        n_samples = 100
        self.documents = pd.DataFrame.from_dict(
            {
                "title": ["test"] * n_samples,
                "url": ["http://url.com"] * n_samples,
                "content": ["cool text"] * n_samples,
                "embedding": [get_fake_embedding()] * n_samples,
                "n_tokens": [10] * n_samples,
                "source": ["fake source"] * n_samples,
            }
        )

    def add(self, documents):
        pass

    def get_documents(self, source):
        return self.documents


def test_chatbot_simple(tmp_path, monkeypatch):
    monkeypatch.setattr("buster.chatbot.get_documents_manager_from_extension", lambda filepath: DocumentsMock)
    monkeypatch.setattr("buster.chatbot.get_embedding", lambda x, engine: get_fake_embedding())

    hf_transformers_cfg = ChatbotConfig(
        documents_file=tmp_path / "not_a_real_file.tar.gz",
        unknown_prompt="This doesn't seem to be related to the huggingface library. I am not sure how to answer.",
        embedding_model="text-embedding-ada-002",
        top_k=3,
        thresh=0.7,
        max_words=3000,
        completion_kwargs={
            "temperature": 0,
            "engine": "text-davinci-003",
            "max_tokens": 100,
        },
        response_format="slack",
        text_before_prompt=(
            """You are a slack chatbot assistant answering technical questions about huggingface transformers, a library to train transformers in python.\n"""
            """Make sure to format your answers in Markdown format, including code block and snippets.\n"""
            """Do not include any links to urls or hyperlinks in your answers.\n\n"""
            """Now answer the following question:\n"""
        ),
    )
    chatbot = Chatbot(hf_transformers_cfg)
    answer = chatbot.process_input("What is a transformer?")
    assert isinstance(answer, str)
