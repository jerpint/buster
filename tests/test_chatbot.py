import os
from pathlib import Path

import numpy as np
import pandas as pd

from buster.buster import Buster, BusterConfig
from buster.completers.base import Completer
from buster.documents import DocumentsManager, get_documents_manager_from_extension
from buster.formatter.base import Response

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
DOCUMENTS_FILE = os.path.join(str(TEST_DATA_DIR), "document_embeddings_huggingface_subset.tar.gz")


def get_fake_embedding(length=1536):
    rng = np.random.default_rng()
    return list(rng.random(length, dtype=np.float32))


class MockCompleter(Completer):
    def __init__(self, expected_answer):
        self.expected_answer = expected_answer

    def complete(self):
        return

    def generate_response(self, user_input, documents) -> Response:
        return Response(self.expected_answer)



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

import logging
logging.basicConfig(level=logging.INFO)

def test_chatbot_mock_data(tmp_path, monkeypatch):
    gpt_expected_answer = "this is GPT answer"
    monkeypatch.setattr(Buster, "get_embedding", lambda x, engine: get_fake_embedding())
    monkeypatch.setattr("buster.buster.get_completer", lambda x: MockCompleter(expected_answer=gpt_expected_answer))

    hf_transformers_cfg = BusterConfig(
        unknown_prompt="This doesn't seem to be related to the huggingface library. I am not sure how to answer.",
        embedding_model="text-embedding-ada-002",
        top_k=3,
        thresh=0,
        max_words=3000,
        response_format="slack",
        source="fake source",
        completer_cfg={
            "name": "GPT3",
            "text_before_prompt": (
                """You are a slack chatbot assistant answering technical questions about huggingface transformers, a library to train transformers in python.\n"""
                """Make sure to format your answers in Markdown format, including code block and snippets.\n"""
                """Do not include any links to urls or hyperlinks in your answers.\n\n"""
                """Now answer the following question:\n"""
            ),
            "text_before_documents": "",
            "completion_kwargs": {
                "engine": "text-davinci-003",
                "max_tokens": 200,
                "temperature": None,
                "top_p": None,
                "frequency_penalty": 1,
                "presence_penalty": 1,
            },
        },
    )
    filepath = tmp_path / "not_a_real_file.tar.gz"
    documents = DocumentsMock(filepath)
    buster = Buster(cfg=hf_transformers_cfg, documents=documents)
    answer = buster.process_input("What is a transformer?")
    assert isinstance(answer, str)
    assert answer.startswith(gpt_expected_answer)


def test_chatbot_real_data__chatGPT():
    hf_transformers_cfg = BusterConfig(
        unknown_prompt="I'm sorry, but I am an AI language model trained to assist with questions related to the huggingface transformers library. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with?",
        embedding_model="text-embedding-ada-002",
        top_k=3,
        thresh=0.7,
        max_words=3000,
        response_format="slack",
        completer_cfg={
            "name": "ChatGPT",
            "text_before_prompt": (
                """You are a slack chatbot assistant answering technical questions about huggingface transformers, a library to train transformers in python.\n"""
                """Make sure to format your answers in Markdown format, including code block and snippets.\n"""
                """Do not include any links to urls or hyperlinks in your answers.\n\n"""
                """Now answer the following question:\n"""
            ),
            "text_before_documents": "Only use these documents as reference:\n",
            "completion_kwargs": {
                "model": "gpt-3.5-turbo",
            },
        },
    )
    documents = get_documents_manager_from_extension(DOCUMENTS_FILE)(DOCUMENTS_FILE)
    buster = Buster(cfg=hf_transformers_cfg, documents=documents)
    answer = buster.process_input("What is a transformer?")
    assert isinstance(answer, str)


def test_chatbot_real_data__chatGPT_OOD():
    buster_cfg = BusterConfig(
        unknown_prompt="I'm sorry, but I am an AI language model trained to assist with questions related to the huggingface transformers library. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with?",
        embedding_model="text-embedding-ada-002",
        top_k=3,
        thresh=0.7,
        max_words=3000,
        completer_cfg={
            "name": "ChatGPT",
            "text_before_prompt": (
                """You are a slack chatbot assistant answering technical questions about huggingface transformers, a library to train transformers in python. """
                """Make sure to format your answers in Markdown format, including code block and snippets. """
                """Do not include any links to urls or hyperlinks in your answers. """
                """If you do not know the answer to a question, or if it is completely irrelevant to the library usage, let the user know you cannot answer. """
                """Use this response: """
                """'I'm sorry, but I am an AI language model trained to assist with questions related to the huggingface transformers library. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with?'\n"""
                """For example:\n"""
                """What is the meaning of life for huggingface?\n"""
                """I'm sorry, but I am an AI language model trained to assist with questions related to the huggingface transformers library. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with?"""
                """Now answer the following question:\n"""
            ),
            "text_before_documents": "Only use these documents as reference:\n",
            "completion_kwargs": {
                "model": "gpt-3.5-turbo",
            },
        },
        response_format="gradio",
    )
    documents = get_documents_manager_from_extension(DOCUMENTS_FILE)(DOCUMENTS_FILE)
    buster = Buster(cfg=buster_cfg, documents=documents)
    answer = buster.process_input("What is a good recipe for brocolli soup?")
    assert isinstance(answer, str)
    assert buster_cfg.unknown_prompt in answer


def test_chatbot_real_data__GPT():
    hf_transformers_cfg = BusterConfig(
        unknown_prompt="This doesn't seem to be related to the huggingface library. I am not sure how to answer.",
        embedding_model="text-embedding-ada-002",
        top_k=3,
        thresh=0.7,
        max_words=3000,
        response_format="slack",
        completer_cfg={
            "name": "GPT3",
            "text_before_prompt": (
                """You are a slack chatbot assistant answering technical questions about huggingface transformers, a library to train transformers in python.\n"""
                """Make sure to format your answers in Markdown format, including code block and snippets.\n"""
                """Do not include any links to urls or hyperlinks in your answers.\n\n"""
                """Now answer the following question:\n"""
            ),
            "text_before_documents": "",
            "completion_kwargs": {
                "engine": "text-davinci-003",
                "max_tokens": 200,
                "temperature": None,
                "top_p": None,
                "frequency_penalty": 1,
                "presence_penalty": 1,
            },
        },
    )
    documents = get_documents_manager_from_extension(DOCUMENTS_FILE)(DOCUMENTS_FILE)
    buster = Buster(cfg=hf_transformers_cfg, documents=documents)
    answer = buster.process_input("What is a transformer?")
    assert isinstance(answer, str)
