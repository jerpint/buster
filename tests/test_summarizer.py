from buster.summarizer.chunks import recompose_chunks, summarize_chunks
from buster.summarizer.utils import resize_chunks, split_string, word_count


def test_word_count():
    # list of strings
    my_list = ["a sentence with words", "another sentence with more words"]
    assert word_count(my_list) == 4 + 5

    # empty list and string
    assert word_count("") == 0
    assert word_count([]) == 0

    my_list = ["", "another sentence with more words"]
    assert word_count(my_list) == 5

    # extra whitespaces
    my_text = " leading whitespace"
    assert (word_count(my_text)) == 2

    my_text = "   more   whitespace "
    assert (word_count(my_text)) == 2

    my_text = "trailing whitespace "
    assert (word_count(my_text)) == 2

    my_list = [" leading whitespace", "trailing whitespace "]
    assert (word_count(my_list)) == 4


def test_split_string__no_split():
    # we don't expect a split because word_count(string) < max_words
    string = "This is a string with much less words than max_words."

    result = split_string(string, max_words=100)

    assert isinstance(result, str)
    assert result == string


def test_split_string_split():
    string = "This is a string that we want to split to chunks."

    results = split_string(string, max_words=4)

    assert isinstance(results, list)
    assert len(results) == 3
    assert results[0] == "This is a string"
    assert results[1] == "that we want to"
    assert results[2] == "split to chunks."


def test_resize_chunks():
    chunks = ["chunk 1", "chunk 2", "another chunk with five words"]
    max_words = 3

    resized_chunks = resize_chunks(chunks, max_words=max_words)

    assert resized_chunks == ["chunk 1", "chunk 2", "another chunk with", "five words"]


# def test_recompose_chunks():
#     chunks = ["Two words", "Three word chunk", "Chunk with four words"]

#     recompose_chunks(chunks, max_words=3)
