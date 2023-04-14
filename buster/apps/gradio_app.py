import logging
import os

import gradio as gr
import pandas as pd

from buster.apps.bot_configs import available_configs
from buster.busterbot import Buster, BusterConfig
from buster.retriever import ServiceRetriever

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_CONFIG = "huggingface"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
retriever = ServiceRetriever(PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX, MONGO_URI, MONGO_DB)

# initialize buster with the default config...
default_cfg: BusterConfig = available_configs.get(DEFAULT_CONFIG)
buster = Buster(cfg=default_cfg, retriever=retriever)


def format_sources(matched_documents: pd.DataFrame) -> str:
    if len(matched_documents) == 0:
        return ""

    sourced_answer_template: str = (
        """📝 Here are the sources I used to answer your question:<br>""" """{sources}<br><br>""" """{footnote}"""
    )
    source_template: str = """[🔗 {source.title}]({source.url}), relevance: {source.similarity:2.1f} %"""

    matched_documents.similarity = matched_documents.similarity * 100
    sources = "<br>".join([source_template.format(source=source) for _, source in matched_documents.iterrows()])
    footnote: str = "I'm a bot 🤖 and not always perfect."

    return sourced_answer_template.format(sources=sources, footnote=footnote)


def chat(question, history, bot_source):
    history = history or []
    cfg = available_configs.get(bot_source)
    buster.update_cfg(cfg)

    response = buster.process_input(question)

    # formatted_sources = source_formatter(sources)
    matched_documents = response.matched_documents

    formatted_sources = format_sources(matched_documents)
    formatted_response = f"{response.completion.text}<br><br>" + formatted_sources

    history.append((question, formatted_response))

    return history, history


block = gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>Buster 🤖: A Question-Answering Bot for open-source libraries </center></h3>")

    doc_source = gr.Dropdown(
        choices=sorted(list(available_configs.keys())),
        value=DEFAULT_CONFIG,
        interactive=True,
        multiselect=False,
        label="Source of Documentation",
        info="The source of documentation to select from",
    )

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="What kind of model should I use for sentiment analysis?",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    examples = gr.Examples(
        # TODO: seems not possible (for now) to update examples on change...
        examples=[
            "What kind of models should I use for images and text?",
            "When should I finetune a model vs. training it form scratch?",
            "Can you give me some python code to quickly finetune a model on my sentiment analysis dataset?",
        ],
        inputs=message,
    )

    gr.Markdown(
        """This simple application uses GPT to search the huggingface 🤗 transformers docs and answer questions.
    For more info on huggingface transformers view the [full documentation.](https://huggingface.co/docs/transformers/index)."""
    )

    gr.HTML("️<center> Created with ❤️ by @jerpint and @hadrienbertrand")

    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[message, state, doc_source], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, state, doc_source], outputs=[chatbot, state])


block.launch(debug=True)
