from dataclasses import dataclass
from typing import Iterable

from buster.formatter import ResponseFormatter, Source


@dataclass
class GradioResponseFormatter(ResponseFormatter):
    """Format the answer for gradio chat interface."""

    error_msg_template: str = """Something went wrong:<br>{response.error_msg}"""
    error_fallback_template: str = "Something went very wrong."
    sourced_answer_template: str = (
        """{response.text}<br><br>"""
        """📝 Here are the sources I used to answer your question:<br>"""
        """{sources}<br><br>"""
        """{footnote}"""
    )
    unsourced_answer_template: str = "{response.text}<br><br>{footnote}"
    source_template: str = """[🔗 {source.title}]({source.url}), relevance: {source.question_similarity:2.1f} %"""

    def sources_list(self, sources: Iterable[Source]) -> str | None:
        """Format sources into a list."""
        items = [self.source_item(source) for source in sources]
        if not items:
            return None  # No list needed.

        return "<br>".join(items)
