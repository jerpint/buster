import json

import pandas as pd
import pytest

from buster.formatters.documents import DocumentsFormatterHTML, DocumentsFormatterJSON
from buster.formatters.prompts import PromptFormatter
from buster.tokenizers import GPTTokenizer


def test_DocumentsDormatterHTML__simple():
    """In this test, we expect all 3 documents to be matched and returned normally."""
    tokenizer = GPTTokenizer(model_name="gpt-3.5-turbo")
    documents_formatter = DocumentsFormatterHTML(
        tokenizer=tokenizer,
        max_tokens=100,
    )

    document_1 = "This is a very short document."
    document_2 = "This is another very short document."
    document_3 = "This is also a short document."

    expected_docs_str = (
        "<DOCUMENTS>"
        f"<DOCUMENT>{document_1}<\\DOCUMENT>"
        f"<DOCUMENT>{document_2}<\\DOCUMENT>"
        f"<DOCUMENT>{document_3}<\\DOCUMENT>"
        "<\\DOCUMENTS>"
    )

    matched_documents = pd.DataFrame({"content": [document_1, document_2, document_3]})

    docs_str, matched_documents_new = documents_formatter.format(matched_documents)

    # less documents and the new document is shorter than the original
    assert all(matched_documents.content == matched_documents_new.content)

    assert docs_str == expected_docs_str


def test_DocumentsDormatterJSON__simple():
    """In this test, we expect all 3 documents to be matched and returned normally."""
    tokenizer = GPTTokenizer(model_name="gpt-3.5-turbo")
    documents_formatter = DocumentsFormatterJSON(tokenizer=tokenizer, max_tokens=100, columns=["content", "source"])

    document_1 = "This is a very short document."
    document_2 = "This is another very short document."
    document_3 = "This is also a short document."

    source_1 = "source 1"
    source_2 = "source 2"
    source_3 = "source 3"

    data_dict = {
        "content": [document_1, document_2, document_3],
        "source": [source_1, source_2, source_3],
    }

    expected_docs_str = json.dumps(
        [
            {"content": document_1, "source": source_1},
            {"content": document_2, "source": source_2},
            {"content": document_3, "source": source_3},
        ],
        separators=(",", ":"),
    )

    matched_documents = pd.DataFrame(data_dict)

    docs_str, matched_documents_new = documents_formatter.format(matched_documents)

    # less documents and the new document is shorter than the original
    assert all(matched_documents.content == matched_documents_new.content)

    assert docs_str == expected_docs_str  # matched_documents.to_json(orient="records")


def test_DocumentsFormatterHTML__doc_to_long():
    """In this test, document_1 doesn't entirely fit.

    we only expect a part of it to be contained.
    """
    tokenizer = GPTTokenizer(model_name="gpt-3.5-turbo")
    documents_formatter = DocumentsFormatterHTML(
        tokenizer=tokenizer,
        max_tokens=100,
    )

    long_sentence = "This is a very long document. It is long on purpose."
    document_1 = long_sentence * 50
    document_2 = "This is a very short document."
    document_3 = "This is also a short document"

    matched_documents = pd.DataFrame({"content": [document_1, document_2, document_3]})

    docs_str, matched_documents_new = documents_formatter.format(matched_documents)

    # less documents and the new document is shorter than the original
    assert len(matched_documents) == 3
    assert len(matched_documents_new) == 1
    assert len(docs_str) < len(document_1)

    # The long document gets truncated, the others don't make it in.
    assert long_sentence in docs_str
    assert document_2 not in docs_str
    assert document_3 not in docs_str


def test_DocumentsFormatterJSON__doc_too_long():
    """In this test, document_3 doesn't fit.
    We expect it to be excluded completely.

    we only expect a part of it to be contained.
    """
    tokenizer = GPTTokenizer(model_name="gpt-3.5-turbo")
    documents_formatter = DocumentsFormatterJSON(tokenizer=tokenizer, max_tokens=100, columns=["content", "source"])

    long_sentence = "This is a very long document. It is long on purpose."

    document_1 = "This is a very short document."
    document_2 = "This is also a short document"
    document_3 = long_sentence * 50

    source_1 = "source 1"
    source_2 = "source 2"
    source_3 = "source 3"

    data_dict = {
        "content": [document_1, document_2, document_3],
        "source": [source_1, source_2, source_3],
    }

    expected_docs_str = json.dumps(
        [
            {"content": document_1, "source": source_1},
            {"content": document_2, "source": source_2},
        ],
        separators=(",", ":"),
    )

    matched_documents = pd.DataFrame(data_dict)

    docs_str, matched_documents_new = documents_formatter.format(matched_documents)
    assert docs_str == expected_docs_str

    # less documents and the new document is shorter than the original
    assert len(matched_documents) == 3
    assert len(matched_documents_new) == 2

    # The last document gets ignored completely, the first 2 make it
    assert document_1 in docs_str
    assert document_2 in docs_str
    assert long_sentence not in docs_str


def test_DocumentsFormatterHTML__doc_to_long_2():
    """In this test, document_2 doesn't entirely fit.

    we only expect a part of it to be contained, as well as all of document_1, and none of document_3.
    """

    tokenizer = GPTTokenizer(model_name="gpt-3.5-turbo")
    documents_formatter = DocumentsFormatterHTML(
        tokenizer=tokenizer,
        max_tokens=100,
    )

    document_1 = "This is a very short document."
    document_2 = "This is a very long document. It is long on purpose." * 50
    document_3 = "This is also a short document"

    matched_documents = pd.DataFrame({"content": [document_1, document_2, document_3]})

    docs_str, matched_documents_new = documents_formatter.format(matched_documents)

    # less documents and the new document is shorter than the original
    assert len(matched_documents) == 3
    assert len(matched_documents_new) == 2

    assert document_1 in docs_str
    assert "This is a very long document. It is long on purpose." in docs_str  # at least a subset should be in there
    assert document_3 not in docs_str


def test_DocumentsFormatterHTML__complex_format():
    """In this test, we expect all 3 documents to be matched and returned in a particular format."""
    tokenizer = GPTTokenizer(model_name="gpt-3.5-turbo")
    documents_formatter = DocumentsFormatterHTML(
        tokenizer=tokenizer,
        max_tokens=100,
        formatter="Title: {title}\n{content}\n",
    )

    document_1 = "This is a very short document."
    document_2 = "This is another very short document."
    document_3 = "This is also a short document."

    title_1 = "doc1"
    title_2 = "doc2"
    title_3 = "doc3"

    country_1 = "Canada"
    country_2 = "France"
    country_3 = "Germany"

    expected_docs_str = (
        "<DOCUMENTS>"
        f"<DOCUMENT>Title: {title_1}\n{document_1}\n<\\DOCUMENT>"
        f"<DOCUMENT>Title: {title_2}\n{document_2}\n<\\DOCUMENT>"
        f"<DOCUMENT>Title: {title_3}\n{document_3}\n<\\DOCUMENT>"
        "<\\DOCUMENTS>"
    )

    matched_documents = pd.DataFrame(
        {
            "content": [document_1, document_2, document_3],
            "title": [title_1, title_2, title_3],
            "country": [country_1, country_2, country_3],
        }
    )

    docs_str, matched_documents_new = documents_formatter.format(matched_documents)

    # less documents and the new document is shorter than the original
    assert all(matched_documents.content == matched_documents_new.content)

    assert docs_str == expected_docs_str


def test_system_prompt_formatter():
    tokenizer = GPTTokenizer(model_name="gpt-3.5-turbo")
    prompt_formatter = PromptFormatter(
        tokenizer=tokenizer,
        max_tokens=200,
        text_after_docs="After docs.",
        text_before_docs="Before docs.",
        formatter="{text_before_docs}\n{documents}\n{text_after_docs}",
    )

    documents = "Here are some docs"

    prompt = prompt_formatter.format(documents)

    assert prompt == ("Before docs.\n" "Here are some docs\n" "After docs.")

    assert documents in prompt


def test_system_prompt_formatter__to_long():
    tokenizer = GPTTokenizer(model_name="gpt-3.5-turbo")
    prompt_formatter = PromptFormatter(
        tokenizer=tokenizer,
        max_tokens=200,
        text_after_docs="After docs.",
        text_before_docs="Before docs.",
    )

    documents = "Here are some documents that are WAY too long." * 100

    with pytest.raises(ValueError):
        prompt_formatter.format(documents)
