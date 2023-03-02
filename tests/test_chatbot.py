import os
from pathlib import Path

from buster.chatbot import Buster, BusterConfig

TEST_DATA_DIR = Path(__file__).resolve().parent / "data"
DOCUMENTS_FILE = os.path.join(str(TEST_DATA_DIR), "document_embeddings_huggingface_subset.tar.gz")


def test_chatbot_simple():
    hf_transformers_cfg = BusterConfig(
        documents_file=DOCUMENTS_FILE,
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
    chatbot = Buster(hf_transformers_cfg)
    answer = chatbot.process_input("What is a transformer?")
    assert isinstance(answer, str)
