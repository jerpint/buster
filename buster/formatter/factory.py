import logging

from buster.formatter import (
    ResponseFormatter,
    HTMLResponseFormatter,
    MarkdownResponseFormatter,
    SlackResponseFormatter,
    GradioResponseFormatter,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ResponseFormatterFactory:
    def get_formatter(self, format, **kwargs):
        logger.info(f"Using formatter: {format}")
        if format == "text":
            return ResponseFormatter(**kwargs)
        elif format == "slack":
            return SlackResponseFormatter(**kwargs)
        elif format == "HTML":
            return HTMLResponseFormatter(**kwargs)
        elif format == "gradio":
            return GradioResponseFormatter(**kwargs)
        elif format == "markdown":
            return MarkdowResponseFormatter(**kwargs)
        else:
            raise ValueError(f"Undefined {format=}")
