import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def word_count(string: str | list[str]):
    """Count the number of words contained in a string or in a list of strings."""
    if isinstance(string, list):
        return word_count(" ".join(string))
    return len(string.split())


def split_string(string: str, max_words: int) -> str | list[str]:
    """Naively split a string into substrings by separating them into segments of length max_words."""
    if len(string) <= max_words:
        return string
    words = string.split()
    split_strings = []
    for i in range(0, len(words), max_words):
        split_strings.append(" ".join(words[i : i + max_words]))

    logger.info(f"split {string} to {split_strings}")
    return split_strings


def resize_chunks(chunks: list[str], max_words: int) -> list[str]:
    """Given a list of chunks(str), if a chunk has more words than max_words, chunks it into separate chunks of equal length of max_words."""
    return [string for chunk in chunks for string in split_string(chunk, max_words=max_words)]
