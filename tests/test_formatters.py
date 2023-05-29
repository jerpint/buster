import pandas as pd
import pytest

from buster.formatters.documents import DocumentsFormatter
from buster.formatters.prompts import PromptFormatter
from buster.tokenizers import GPTTokenizer


def test_documents_formatter__normal():
    """In this test, we expect all 3 documents to be matched and returned normally."""
    tokenizer = GPTTokenizer(model_name="gpt-3.5-turbo")
    documents_formatter = DocumentsFormatter(
        tokenizer=tokenizer,
        max_tokens=100,
    )

    document_1 = "This is a very short document."
    document_2 = "This is another very short document."
    document_3 = "This is also a short document."

    expected_docs_str = (
        f"<DOCUMENT>{document_1}<\\DOCUMENT>"
        f"<DOCUMENT>{document_2}<\\DOCUMENT>"
        f"<DOCUMENT>{document_3}<\\DOCUMENT>"
    )

    matched_documents = pd.DataFrame({"content": [document_1, document_2, document_3]})

    docs_str, matched_documents_new = documents_formatter.format(matched_documents)

    # less documents and the new document is shorter than the original
    assert all(matched_documents.content == matched_documents_new.content)

    assert docs_str == expected_docs_str


def test_documents_formatter__doc_to_long():
    """In this test, document_1 doesn't entirely fit.

    we only expect a part of it to be contained.
    """
    tokenizer = GPTTokenizer(model_name="gpt-3.5-turbo")
    documents_formatter = DocumentsFormatter(
        tokenizer=tokenizer,
        max_tokens=100,
    )

    document_1 = "This is a very long document. It is long on purpose." * 50
    document_2 = "This is a very short document."
    document_3 = "This is also a short document"

    matched_documents = pd.DataFrame({"content": [document_1, document_2, document_3]})

    docs_str, matched_documents_new = documents_formatter.format(matched_documents)

    # less documents and the new document is shorter than the original
    assert len(matched_documents) == 3
    assert len(matched_documents_new) == 1
    assert len(docs_str) < len(document_1)

    # ignore the <DOCUMENT> </DOCUMENT> tags from the formatter...
    assert docs_str[11:-11] in document_1


def test_documents_formatter__doc_to_long_2():
    """In this test, document_2 doesn't entirely fit.

    we only expect a part of it to be contained, as well as all of document_1, and none of document_3.
    """

    tokenizer = GPTTokenizer(model_name="gpt-3.5-turbo")
    documents_formatter = DocumentsFormatter(
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


def test_documents_formatter__complex_format():
    """In this test, we expect all 3 documents to be matched and returned in a particular format."""
    tokenizer = GPTTokenizer(model_name="gpt-3.5-turbo")
    documents_formatter = DocumentsFormatter(
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
        f"<DOCUMENT>Title: {title_1}\n{document_1}\n<\\DOCUMENT>"
        f"<DOCUMENT>Title: {title_2}\n{document_2}\n<\\DOCUMENT>"
        f"<DOCUMENT>Title: {title_3}\n{document_3}\n<\\DOCUMENT>"
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
