import logging

import buster.formatter as F

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def response_formatter_factory(format: str, **kwargs):
    pass
    # logger.info(f"Using formatter: {format}")
    # if format == "text":
    #     return F.ResponseFormatter(**kwargs)
    # elif format == "slack":
    #     return F.SlackResponseFormatter(**kwargs)
    # elif format == "HTML":
    #     return F.HTMLResponseFormatter(**kwargs)
    # elif format == "gradio":
    #     return F.GradioResponseFormatter(**kwargs)
    # elif format == "markdown":
    #     return F.MarkdownResponseFormatter(**kwargs)
    # else:
    #     raise ValueError(f"Undefined {format=}")
