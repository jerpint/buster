import copy
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buster.busterbot import Buster, BusterConfig
from buster.completers import ChatGPTCompleter, Completer, Completion, DocumentAnswerer
from buster.documents_manager import DeepLakeDocumentsManager
from buster.formatters.documents import DocumentsFormatter
from buster.formatters.prompts import PromptFormatter
from buster.retriever import DeepLakeRetriever, Retriever
from buster.tokenizers.gpt import GPTTokenizer
from buster.validators import QuestionAnswerValidator, Validator

logging.basicConfig(level=logging.INFO)


DOCUMENTS_CSV = Path(__file__).resolve().parent.parent / "buster/examples/stackoverflow.csv"
UNKNOWN_PROMPT = "I'm sorry but I don't know how to answer."
NUM_WORKERS = 1

# default class used by our tests
buster_cfg_template = BusterConfig(
    completion_cfg={
        "completion_kwargs": {
            "model": "gpt-3.5-turbo",
            "temperature": 0,
        },
    },
    validator_cfg={
        "unknown_response_templates": [
            UNKNOWN_PROMPT,
        ],
        "unknown_threshold": 0.85,
        "embedding_model": "text-embedding-ada-002",
        "use_reranking": True,
        "check_question_prompt": "You are validating if questions are related to AI. If a question is relevant, respond with 'true', if it is irrlevant, respond with 'false'.",
        "completion_kwargs": {"temperature": 0, "model": "gpt-3.5-turbo"},
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


class MockAnswerer(Completer):
    def __init__(self, expected_answer):
        self.expected_answer = expected_answer

    def prepare_prompt(self, user_input, matched_documents):
        pass

    def complete(self):
        return

    def get_completion(self, user_input, matched_documents, validator, *arg, **kwarg) -> Completion:
        return Completion(
            answer_text=self.expected_answer,
            error=False,
            user_input=user_input,
            matched_documents=matched_documents,
            validator=validator,
        )


class MockRetriever(Retriever):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        path = kwargs["path"]

        self.path = path

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

    def get_topk_documents(self, query: str, source: str = None, top_k: int = None) -> pd.DataFrame:
        documents = self.documents
        documents["embedding"] = [get_fake_embedding() for _ in range(len(documents))]
        documents["similarity"] = [np.random.random() for _ in range(len(documents))]
        return documents

    def get_embedding(self, query, engine):
        return get_fake_embedding()

    def get_source_display_name(self, source):
        return source


class MockValidator(Validator):
    def __init__(self, *args, **kwargs):
        return

    def validate(self, completion):
        completion.answer_relevant = True
        return completion

    def check_question_relevance(self, *args, **kwargs):
        return True, ""

    def check_answer_relevance(self, *args, **kwargs):
        return True


@pytest.fixture(scope="session")
def vector_store_path(tmp_path_factory):
    # Create a temporary directory and folder for the database manager
    dm_path = tmp_path_factory.mktemp("data").joinpath("deeplake_store")

    # Add the documents (will generate embeddings)
    dm = DeepLakeDocumentsManager(vector_store_path=dm_path)
    df = pd.read_csv(DOCUMENTS_CSV)
    dm.add(df, num_workers=NUM_WORKERS)
    return dm_path


def test_chatbot_mock_data(tmp_path, monkeypatch):
    gpt_expected_answer = "this is GPT answer"

    path = tmp_path / "not_a_real_file.tar.gz"

    buster_cfg = copy.deepcopy(buster_cfg_template)
    buster_cfg.retriever_cfg["path"] = path
    buster_cfg.completion_cfg = {
        "expected_answer": gpt_expected_answer,
    }

    retriever = MockRetriever(**buster_cfg.retriever_cfg)
    document_answerer = MockAnswerer(**buster_cfg.completion_cfg)
    validator = MockValidator(**buster_cfg.validator_cfg)
    buster = Buster(retriever=retriever, document_answerer=document_answerer, validator=validator)
    completion = buster.process_input("What is a transformer?", source="fake_source")
    assert isinstance(completion.answer_text, str)
    assert completion.answer_text.startswith(gpt_expected_answer)


def test_chatbot_real_data__chatGPT(vector_store_path):
    buster_cfg = copy.deepcopy(buster_cfg_template)
    buster_cfg.retriever_cfg["path"] = vector_store_path

    retriever: Retriever = DeepLakeRetriever(**buster_cfg.retriever_cfg)
    tokenizer = GPTTokenizer(**buster_cfg.tokenizer_cfg)
    document_answerer = DocumentAnswerer(
        completer=ChatGPTCompleter(**buster_cfg.completion_cfg),
        documents_formatter=DocumentsFormatter(tokenizer=tokenizer, **buster_cfg.documents_formatter_cfg),
        prompt_formatter=PromptFormatter(tokenizer=tokenizer, **buster_cfg.prompt_formatter_cfg),
    )
    validator: Validator = QuestionAnswerValidator(**buster_cfg.validator_cfg)
    buster: Buster = Buster(retriever=retriever, document_answerer=document_answerer, validator=validator)

    completion = buster.process_input("What is backpropagation?")
    assert isinstance(completion.answer_text, str)

    assert completion.question_relevant == True
    assert completion.answer_relevant == True

    assert completion.completion_kwargs == buster_cfg.completion_cfg["completion_kwargs"]


def test_chatbot_real_data__chatGPT_OOD(vector_store_path):
    buster_cfg = copy.deepcopy(buster_cfg_template)
    buster_cfg.retriever_cfg["path"] = vector_store_path
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

    retriever: Retriever = DeepLakeRetriever(**buster_cfg.retriever_cfg)
    tokenizer = GPTTokenizer(**buster_cfg.tokenizer_cfg)
    document_answerer = DocumentAnswerer(
        completer=ChatGPTCompleter(**buster_cfg.completion_cfg),
        documents_formatter=DocumentsFormatter(tokenizer=tokenizer, **buster_cfg.documents_formatter_cfg),
        prompt_formatter=PromptFormatter(tokenizer=tokenizer, **buster_cfg.prompt_formatter_cfg),
    )
    validator: Validator = QuestionAnswerValidator(**buster_cfg.validator_cfg)
    buster: Buster = Buster(retriever=retriever, document_answerer=document_answerer, validator=validator)

    completion: Completion = buster.process_input("What is a good recipe for brocolli soup?")
    assert isinstance(completion.answer_text, str)

    assert completion.question_relevant == False
    assert completion.answer_relevant == False

    assert completion.completion_kwargs is None


def test_chatbot_real_data__no_docs_found(vector_store_path):
    with pytest.warns():
        buster_cfg = copy.deepcopy(buster_cfg_template)
        buster_cfg.retriever_cfg = {
            "path": vector_store_path,
            "embedding_model": "text-embedding-ada-002",
            "top_k": 3,
            "thresh": 1,  # Set threshold very high to be sure no docs are matched
            "max_tokens": 3000,
        }
        buster_cfg.documents_answerer_cfg["no_documents_message"] = "No documents available."
        retriever: Retriever = DeepLakeRetriever(**buster_cfg.retriever_cfg)
        tokenizer = GPTTokenizer(**buster_cfg.tokenizer_cfg)
        document_answerer = DocumentAnswerer(
            completer=ChatGPTCompleter(**buster_cfg.completion_cfg),
            documents_formatter=DocumentsFormatter(tokenizer=tokenizer, **buster_cfg.documents_formatter_cfg),
            prompt_formatter=PromptFormatter(tokenizer=tokenizer, **buster_cfg.prompt_formatter_cfg),
            **buster_cfg.documents_answerer_cfg,
        )
        validator: Validator = QuestionAnswerValidator(**buster_cfg.validator_cfg)
        buster: Buster = Buster(retriever=retriever, document_answerer=document_answerer, validator=validator)

        completion = buster.process_input("What is backpropagation?")
        assert isinstance(completion.answer_text, str)

        assert completion.question_relevant == True
        assert completion.answer_relevant == False
        assert completion.answer_text == "No documents available."
