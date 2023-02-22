from dataclasses import dataclass
import html
from typing import Iterable
from buster.formatter.base import Formatter, Response, Source


@dataclass
class HTMLFormatter(Formatter):
    """Format the answer in HTML."""

    source_template: str = """<li><a href='{source.url}'>🔗 {source.name}</a></li>"""
    error_msg_template: str = """<div class="error">Something went wrong:\n<p>{response.error_msg}</p></div>"""
    error_fallback_template: str = """<div class="error">Something went very wrong.</div>"""
    sourced_answer_template: str = (
        """<div class="answer"><p>{response.text}</p></div>\n"""
        """<div class="sources>📝 Here are the sources I used to answer your question:\n"""
        """<ol>\n{sources}</ol></div>\n"""
        """<div class="footer">I'm a chatbot, bleep bloop.</div>"""
    )
    unsourced_answer_template: str = (
        """<div class="answer">{response.text}</div>\n<div class="footer">I'm a chatbot, bleep bloop.</div>"""
    )

    def sources_list(self, sources: Iterable[Source]) -> str | None:
        """Format sources into a list."""
        items = [self.source_item(source) for source in sources]
        if not items:
            return None  # No list needed.

        return "\n".join(items)

    def __call__(self, response: Response, sources: Iterable[Source]) -> str:
        # Escape any html in the text.
        response = Response(
            html.escape(response.text) if response.text else response.text,
            response.error,
            html.escape(response.error_msg) if response.error_msg else response.error_msg,
        )
        sources = (Source(html.escape(source.name), source.url, source.question_similarity) for source in sources)
        return super().__call__(response, sources)
