import copy
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buster.busterbot import Buster, BusterConfig
from buster.completers.base import ChatGPTCompleter, Completer, Completion
from buster.docparser import generate_embeddings
from buster.documents.sqlite.documents import DocumentsDB
from buster.formatters.documents import DocumentsFormatter
from buster.formatters.prompts import PromptFormatter
from buster.retriever import Retriever
from buster.retriever.sqlite import SQLiteRetriever
from buster.tokenizers.gpt import GPTTokenizer
from buster.validators.base import Validator

logging.basicConfig(level=logging.INFO)


DOCUMENTS_CSV = Path(__file__).resolve().parent.parent / "buster/examples/stackoverflow.csv"
UNKNOWN_PROMPT = "I'm sorry but I don't know how to answer."

# default class used by our tests
buster_cfg_template = BusterConfig(
    completion_cfg={
        "completion_kwargs": {
            "model": "gpt-3.5-turbo",
            "temperature": 0,
        },
    },
    validator_cfg={
        "unknown_prompt": UNKNOWN_PROMPT,
        "unknown_threshold": 0.85,
        "embedding_model": "text-embedding-ada-002",
        "use_reranking": True,
    },
    retriever_cfg={
        # "db_path": to be set using pytest fixture,
        "top_k": 3,
        "thresh": 0.7,
        "max_tokens": 2000,
        "embedding_model": "text-embedding-ada-002",
    },
    prompt_formatter_cfg={
        "max_tokens": 3500,
        "text_after_docs": ("""Now answer the following question:\n"""),
        "text_before_docs": (
            """You are a chatbot assistant answering technical questions about artificial intelligence (AI). """
            """If you do not know the answer to a question, or if it is completely irrelevant to your domain knowledge of AI library usage, let the user know you cannot answer."""
            """Use this response when you cannot answer:\n"""
            f"""'{UNKNOWN_PROMPT}'\n"""
            """For example:\n"""
            """What is the meaning of life?\n"""
            f"""'{UNKNOWN_PROMPT}'\n"""
            """Only use these prodived documents as reference:\n"""
        ),
    },
    documents_formatter_cfg={
        "max_tokens": 3000,
        "formatter": "{content}",
    },
)


def get_fake_embedding(length=1536):
    rng = np.random.default_rng()
    return list(rng.random(length, dtype=np.float32))


class MockCompleter(Completer):
    def __init__(self, expected_answer):
        self.expected_answer = expected_answer

    def prepare_prompt(self, user_input, matched_documents):
        pass

    def complete(self):
        return

    def get_completion(self, user_input, matched_documents) -> Completion:
        return Completion(
            completor=self.expected_answer, error=False, user_input=user_input, matched_documents=matched_documents
        )


class MockRetriever(Retriever):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        db_path = kwargs["db_path"]

        self.db_path = db_path

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

    def get_documents(self, source):
        return self.documents

    def get_embedding(self, query, engine):
        return get_fake_embedding()

    def get_source_display_name(self, source):
        return source


class MockValidator(Validator):
    def validate(self, completion):
        completion.answer_relevant = True
        return completion


@pytest.fixture(scope="session")
def database_file(tmp_path_factory):
    # Create a temporary directory and file for the database
    db_file = tmp_path_factory.mktemp("data").joinpath("documents.db")

    # Generate the actual embeddings
    documents_manager = DocumentsDB(db_file)
    documents = pd.read_csv(DOCUMENTS_CSV)
    documents = generate_embeddings(documents, documents_manager)
    yield db_file

    # Teardown: Remove the temporary database file
    db_file.unlink()


def test_chatbot_mock_data(tmp_path, monkeypatch):
    gpt_expected_answer = "this is GPT answer"

    db_path = tmp_path / "not_a_real_file.tar.gz"

    buster_cfg = copy.deepcopy(buster_cfg_template)
    buster_cfg.retriever_cfg["db_path"] = db_path
    buster_cfg.completion_cfg = {
        "expected_answer": gpt_expected_answer,
    }

    retriever = MockRetriever(**buster_cfg.retriever_cfg)
    completer = MockCompleter(**buster_cfg.completion_cfg)
    validator = MockValidator(**buster_cfg.validator_cfg)
    buster = Buster(retriever=retriever, completer=completer, validator=validator)
    completion = buster.process_input("What is a transformer?", source="fake_source")
    assert isinstance(completion.text, str)
    assert completion.text.startswith(gpt_expected_answer)


def test_chatbot_real_data__chatGPT(database_file):
    buster_cfg = copy.deepcopy(buster_cfg_template)
    buster_cfg.retriever_cfg["db_path"] = database_file

    retriever: Retriever = SQLiteRetriever(**buster_cfg.retriever_cfg)
    tokenizer = GPTTokenizer(**buster_cfg.tokenizer_cfg)
    completer: Completer = ChatGPTCompleter(
        documents_formatter=DocumentsFormatter(tokenizer=tokenizer, **buster_cfg.documents_formatter_cfg),
        prompt_formatter=PromptFormatter(tokenizer=tokenizer, **buster_cfg.prompt_formatter_cfg),
        **buster_cfg.completion_cfg,
    )
    validator: Validator = Validator(**buster_cfg.validator_cfg)
    buster: Buster = Buster(retriever=retriever, completer=completer, validator=validator)

    completion = buster.process_input("What is backpropagation?")
    completion = buster.postprocess_completion(completion)
    assert isinstance(completion.text, str)

    assert completion.answer_relevant == True


def test_chatbot_real_data__chatGPT_OOD(database_file):
    buster_cfg = copy.deepcopy(buster_cfg_template)
    buster_cfg.retriever_cfg["db_path"] = database_file
    buster_cfg.prompt_formatter_cfg = {
        "max_tokens": 3500,
        "text_before_docs": (
            """You are a chatbot assistant answering technical questions about artificial intelligence (AI)."""
            """If you do not know the answer to a question, or if it is completely irrelevant to your domain knowledge of AI library usage, let the user know you cannot answer."""
            """Use this response: """
            f"""'{UNKNOWN_PROMPT}'\n"""
            """For example:\n"""
            """What is the meaning of life?\n"""
            f"""'{UNKNOWN_PROMPT}'\n"""
            """Now answer the following question:\n"""
        ),
        "text_after_docs": "Only use these documents as reference:\n",
    }

    retriever: Retriever = SQLiteRetriever(**buster_cfg.retriever_cfg)
    tokenizer = GPTTokenizer(**buster_cfg.tokenizer_cfg)
    completer: Completer = ChatGPTCompleter(
        documents_formatter=DocumentsFormatter(tokenizer=tokenizer, **buster_cfg.documents_formatter_cfg),
        prompt_formatter=PromptFormatter(tokenizer=tokenizer, **buster_cfg.prompt_formatter_cfg),
        **buster_cfg.completion_cfg,
    )
    validator: Validator = Validator(**buster_cfg.validator_cfg)
    buster: Buster = Buster(retriever=retriever, completer=completer, validator=validator)

    completion = buster.process_input("What is a good recipe for brocolli soup?")
    completion = buster.postprocess_completion(completion)
    assert isinstance(completion.text, str)

    assert completion.answer_relevant == False


def test_chatbot_real_data__no_docs_found(database_file):
    buster_cfg = copy.deepcopy(buster_cfg_template)
    buster_cfg.retriever_cfg = {
        "db_path": database_file,
        "embedding_model": "text-embedding-ada-002",
        "top_k": 3,
        "thresh": 1,  # Set threshold very high to be sure no docs are matched
        "max_tokens": 3000,
    }
    buster_cfg.completion_cfg["no_documents_message"] = "No documents available."
    retriever: Retriever = SQLiteRetriever(**buster_cfg.retriever_cfg)
    tokenizer = GPTTokenizer(**buster_cfg.tokenizer_cfg)
    completer: Completer = ChatGPTCompleter(
        documents_formatter=DocumentsFormatter(tokenizer=tokenizer, **buster_cfg.documents_formatter_cfg),
        prompt_formatter=PromptFormatter(tokenizer=tokenizer, **buster_cfg.prompt_formatter_cfg),
        **buster_cfg.completion_cfg,
    )
    validator: Validator = Validator(**buster_cfg.validator_cfg)
    buster: Buster = Buster(retriever=retriever, completer=completer, validator=validator)

    completion = buster.process_input("What is backpropagation?")
    completion = buster.postprocess_completion(completion)
    assert isinstance(completion.text, str)

    assert completion.answer_relevant == False
    assert completion.text == "No documents available."
