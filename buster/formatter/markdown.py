from dataclasses import dataclass
from typing import Iterable

from buster.formatter.base import ResponseFormatter, Source


@dataclass
class MarkdownResponseFormatter(ResponseFormatter):
    """Format the answer in markdown."""

    source_template: str = """[🔗 {source.source}]({source.url}), relevance: {source.question_similarity:2.3f}"""

    def sources_list(self, sources: Iterable[Source]) -> str | None:
        """Format sources into a list."""
        items = [self.source_item(source) for source in sources]
        if not items:
            return None  # No list needed.

        return "\n".join(f"{ind}. {item}" for ind, item in enumerate(items, 1))
