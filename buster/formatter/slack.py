from dataclasses import dataclass
from typing import Iterable

from buster.formatter import ResponseFormatter, Source


@dataclass
class SlackResponseFormatter(ResponseFormatter):
    """Format the answer for Slack."""

    source_template: str = """<{source.url}|🔗 {source.title}>, relevance: {source.question_similarity:2.3f}"""

    def sources_list(self, sources: Iterable[Source]) -> str | None:
        """Format sources into a list."""
        items = [self.source_item(source) for source in sources]
        if not items:
            return None  # No list needed.

        return "\n".join(items)
