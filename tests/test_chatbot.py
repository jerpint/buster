import copy
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from buster.busterbot import Buster, BusterAnswer, BusterConfig
from buster.completers.base import ChatGPTCompleter, Completer, Completion
from buster.formatters.documents import documents_formatter_factory
from buster.formatters.prompts import prompt_formatter_factory
from buster.retriever import Retriever
from buster.tokenizers import tokenizer_factory
from buster.utils import get_retriever_from_extension
from buster.validators.base import Validator

logging.basicConfig(level=logging.INFO)


TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = os.path.join(str(TEST_DATA_DIR), "document_embeddings_huggingface_subset.tar.gz")
DB_SOURCE = "huggingface"
UNKNOWN_PROMPT = "I'm sorry but I don't know how to answer."

# default class used by our tests
buster_cfg_template = BusterConfig(
    completion_cfg={
        "name": "ChatGPT",
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
        "db_path": DB_PATH,
        "top_k": 3,
        "thresh": 0.7,
        "max_tokens": 2000,
        "embedding_model": "text-embedding-ada-002",
    },
    prompt_cfg={
        "max_tokens": 3500,
        "text_before_documents": "",
        "text_before_prompt": (
            """You are a slack chatbot assistant answering technical questions about huggingface transformers, a library to train transformers in python.\n"""
            """Make sure to format your answers in Markdown format, including code block and snippets.\n"""
            """Do not include any links to urls or hyperlinks in your answers.\n\n"""
            """Now answer the following question:\n"""
        ),
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

    def generate_response(self, user_input, matched_documents) -> Completion:
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


def test_chatbot_real_data__chatGPT():
    buster_cfg = copy.deepcopy(buster_cfg_template)

    validator = Validator(**buster_cfg.validator_cfg)
    retriever = get_retriever_from_extension(DB_PATH)(**buster_cfg.retriever_cfg)
    tokenizer = tokenizer_factory(buster_cfg.tokenizer_cfg)
    prompt_formatter = prompt_formatter_factory(tokenizer=tokenizer, prompt_cfg=buster_cfg.prompt_cfg)
    documents_formatter = documents_formatter_factory(
        tokenizer=tokenizer,
        max_tokens=3000,
        # TODO: put max tokens somewhere useful
    )
    completer: Completer = ChatGPTCompleter(
        completion_kwargs=buster_cfg.completion_cfg["completion_kwargs"],
        documents_formatter=documents_formatter,
        prompt_formatter=prompt_formatter,
    )
    buster = Buster(retriever=retriever, completer=completer, validator=validator)
    completion = buster.process_input("What is a transformer?", source=DB_SOURCE)
    completion = buster.postprocess_completion(completion)
    assert isinstance(completion.text, str)

    assert completion.answer_relevant == True


def test_chatbot_real_data__chatGPT_OOD():
    buster_cfg = copy.deepcopy(buster_cfg_template)
    buster_cfg.prompt_cfg = {
        "max_tokens": 3500,
        "text_before_prompt": (
            """You are a chatbot assistant answering technical questions about huggingface transformers, a library to train transformers in python. """
            """Make sure to format your answers in Markdown format, including code block and snippets. """
            """Do not include any links to urls or hyperlinks in your answers. """
            """If you do not know the answer to a question, or if it is completely irrelevant to the library usage, let the user know you cannot answer. """
            """Use this response: """
            f"""'{UNKNOWN_PROMPT}'\n"""
            """For example:\n"""
            """What is the meaning of life for huggingface?\n"""
            f"""'{UNKNOWN_PROMPT}'\n"""
            """Now answer the following question:\n"""
        ),
        "text_before_documents": "Only use these documents as reference:\n",
    }

    validator = Validator(**buster_cfg.validator_cfg)
    retriever = get_retriever_from_extension(DB_PATH)(**buster_cfg.retriever_cfg)
    tokenizer = tokenizer_factory(buster_cfg.tokenizer_cfg)
    prompt_formatter = prompt_formatter_factory(tokenizer=tokenizer, prompt_cfg=buster_cfg.prompt_cfg)
    documents_formatter = documents_formatter_factory(
        tokenizer=tokenizer,
        max_tokens=3000,
        # TODO: put max tokens somewhere useful
    )
    completer: Completer = ChatGPTCompleter(
        completion_kwargs=buster_cfg.completion_cfg["completion_kwargs"],
        documents_formatter=documents_formatter,
        prompt_formatter=prompt_formatter,
    )
    buster = Buster(retriever=retriever, completer=completer, validator=validator)

    completion = buster.process_input("What is a good recipe for brocolli soup?", source=DB_SOURCE)
    completion = buster.postprocess_completion(completion)
    assert isinstance(completion.text, str)

    assert completion.answer_relevant == False


def test_chatbot_real_data__no_docs_found():
    buster_cfg = copy.deepcopy(buster_cfg_template)
    buster_cfg.retriever_cfg = {
        "db_path": DB_PATH,
        "embedding_model": "text-embedding-ada-002",
        "top_k": 3,
        "thresh": 1,  # Set threshold very high to be sure no docs are matched
        "max_tokens": 3000,
    }
    validator = Validator(**buster_cfg.validator_cfg)
    retriever = get_retriever_from_extension(DB_PATH)(**buster_cfg.retriever_cfg)
    tokenizer = tokenizer_factory(buster_cfg.tokenizer_cfg)
    prompt_formatter = prompt_formatter_factory(tokenizer=tokenizer, prompt_cfg=buster_cfg.prompt_cfg)
    documents_formatter = documents_formatter_factory(
        tokenizer=tokenizer,
        max_tokens=3000,
        # TODO: put max tokens somewhere useful
    )
    completer: Completer = ChatGPTCompleter(
        completion_kwargs=buster_cfg.completion_cfg["completion_kwargs"],
        documents_formatter=documents_formatter,
        prompt_formatter=prompt_formatter,
        no_documents_message="No documents available.",
    )
    buster = Buster(retriever=retriever, completer=completer, validator=validator)

    completion = buster.process_input("What is a transformer?", source=DB_SOURCE)
    completion = buster.postprocess_completion(completion)
    assert isinstance(completion.text, str)

    assert completion.answer_relevant == False
    assert completion.text == "No documents available."
