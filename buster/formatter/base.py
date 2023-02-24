from dataclasses import dataclass
from typing import Iterable, NamedTuple

# Should be from the `documents` module.
class Source(NamedTuple):
    name: str
    url: str
    question_similarity: float
    # TODO Add answer similarity.
    # answer_similarity: float


# Should be from the `nlp` module.
@dataclass(slots=True)
class Response:
    text: str
    error: bool = False
    error_msg: str | None = None


@dataclass
class Formatter:

    source_template: str = "{source.name} (relevance: {source.question_similarity:2.3f})"
    error_msg_template: str = "Something went wrong: {response.error_msg}"
    error_fallback_template: str = "Something went very wrong."
    sourced_answer_template: str = "{response.text}\n\nSources:\n{sources}\n\nBut what do I know, I'm a chatbot."
    unsourced_answer_template: str = "{response.text}\n\nBut what do I know, I'm a chatbot."

    def source_item(self, source: Source) -> str:
        """Format a single source item."""
        return self.source_template.format(source=source)

    def sources_list(self, sources: Iterable[Source]) -> str | None:
        """Format sources into a list."""
        items = [self.source_item(source) for source in sources]
        if not items:
            return None  # No list needed.

        return "\n".join(f"{ind}. {item}" for ind, item in enumerate(items, 1))

    def error(self, response: Response) -> str:
        """Format an error message."""
        if response.error_msg:
            return self.error_msg_template.format(response=response)
        return self.error_fallback_template.format(response=response)

    def answer(self, response: Response, sources: Iterable[Source]) -> str:
        """Format an answer and its sources."""
        sources_list = self.sources_list(sources)
        if not sources_list:
            return self.sourced_answer_template.format(response=response, sources=sources_list)

        return self.unsourced_answer_template.format(response=response)

    def __call__(self, response: Response, sources: Iterable[Source]) -> str:
        """Format an answer and its sources, or an error message."""
        if response.error:
            return self.error(response)
        return self.answer(response, sources)
