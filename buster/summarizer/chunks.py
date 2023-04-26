import copy
import logging

from .base import Summarizer
from .utils import resize_chunks, word_count

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def summarize_chunks(chunks: list[str], max_words=600):
    """Recursively recompose chunks and summarize them.

    Similar approach to the algorithm proposed in Recursively Summarizing Books with Human Feedback, minus the actual feedback. (https://arxiv.org/abs/2109.10862)
    """

    chunk_summarizer = Summarizer(
        model="gpt-3.5-turbo",
        system_prompt="Summarize the following text into succinct points.",
        temperature=0,
    )

    overall_summarizer = Summarizer(
        model="gpt-3.5-turbo",
        system_prompt="Here are a list of key points from an article, summarize them into a few cohesive sentences.",
        temperature=0,
    )

    logger.info(f"recomposing {len(chunks)} chunks...")
    parsed_chunks = recompose_chunks(chunks, max_words=max_words)
    logger.info(f"Chunks recomposed to {len(parsed_chunks)} chunks...")

    if len(parsed_chunks) == 1:
        # end recursion, we summarize the doc in natural language.
        return overall_summarizer.summarize(parsed_chunks[0])

    else:
        chunk_summaries = []
        for chunk in parsed_chunks:
            # TODO: this can be done async eventually...
            chunk_summaries.append(chunk_summarizer.summarize(chunk))
        return summarize_chunks(chunk_summaries, max_words=max_words)


def recompose_chunks(chunks: list[str], max_words) -> list[str]:
    """Redefines a list of chunks so that they are ~max_words long.

    This uses a simple heuristic: add chunks together until their combined word_count ~= max_words."""
    # check first to see if our current chunk size is small enough to fit for single summarization task
    if word_count(chunks) <= max_words:
        return ["\n".join(chunks)]

    # if a chunk is longer than max_words, cut it into equal length chunks of max_words
    resized_chunks = resize_chunks(chunks)
    if len(resized_chunks) != len(chunks):
        logger.warning(f"Chunks with num_words > {max_words} were split. Consider reviewing how chunks were made.")

    # ...if it isn't, proceed to procedural chunking
    recomposed_chunks = []  # we will keep track of all our "new chunks" here
    current_chunk = []  # store temporary chunks until they're big enough (e.g. enough words)
    for chunk in resized_chunks:
        candidate_chunk = copy.deepcopy(current_chunk)
        candidate_chunk.append(chunk)

        if word_count(candidate_chunk) <= max_words:
            # still room, add the chunk to the current chunk
            current_chunk = candidate_chunk
            continue
        else:
            # no room for the incoming chunk, add the current chunks to the stack
            # and start over with the newest chunk
            recomposed_chunks.append("\n".join(current_chunk))
            current_chunk = [chunk]

    return recomposed_chunks
