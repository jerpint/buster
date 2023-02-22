from dataclasses import dataclass
from typing import Iterable
from buster.formatter.base import Formatter, Source


@dataclass
class MarkdownFormatter(Formatter):
    """Format the answer in markdown."""

    source_template: str = """[🔗 {source.name}]({source.url}), relevance: {source.question_similarity:2.3f}"""
    error_msg_template: str = """Something went wrong:\n{response.error_msg}"""
    error_fallback_template: str = """Something went very wrong."""
    sourced_answer_template: str = (
        """{response.text}\n\n"""
        """📝 Here are the sources I used to answer your question:\n"""
        """{sources}\n\n"""
        """I'm a chatbot, bleep bloop."""
    )
    unsourced_answer_template: str = """{response.text}\n\nI'm a chatbot, bleep bloop."""

    def sources_list(self, sources: Iterable[Source]) -> str | None:
        """Format sources into a list."""
        items = [self.source_item(source) for source in sources]
        if not items:
            return None  # No list needed.

        return "\n".join(f"{ind}. {item}" for ind, item in enumerate(items, 1))
