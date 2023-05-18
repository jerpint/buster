from buster.completers.base import ChatGPTCompleter, Completer
from buster.formatters.documents import document_formatter_factory
from buster.formatters.prompts import prompt_formatter_factory
from buster.tokenizers import tokenizer_factory
import cfg
import gradio as gr
import pandas as pd

from buster.busterbot import Buster
from buster.retriever import Retriever
from buster.validators.base import Validator, validator_factory
from buster.utils import get_retriever_from_extension
from buster.retriever import SQLiteRetriever

# initialize buster with the config in config.py (adapt to your needs) ...
retriever: Retriever = SQLiteRetriever(**cfg.buster_cfg.retriever_cfg)

tokenizer = tokenizer_factory(cfg.buster_cfg.tokenizer_cfg)
prompt_formatter = prompt_formatter_factory(tokenizer=tokenizer, prompt_cfg=cfg.buster_cfg.prompt_cfg)
documents_formatter = document_formatter_factory(
    tokenizer=tokenizer,
    max_tokens=3000,
    # TODO: put max tokens somewhere useful
)
completer: Completer = ChatGPTCompleter(
    completion_kwargs=cfg.buster_cfg.completion_cfg["completion_kwargs"],
    documents_formatter=documents_formatter,
    prompt_formatter=prompt_formatter,
)
validator: Validator = Validator(**cfg.buster_cfg.validator_cfg)
buster: Buster = Buster(cfg=cfg.buster_cfg, retriever=retriever, completer=completer, validator=validator)


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


def add_sources(history, completion):
    completion = buster.postprocess_completion(completion)

    if completion.response_is_relevant:
        formatted_sources = format_sources(completion.matched_documents)
        history.append([None, formatted_sources])

    return history


def user(user_input, history):
    """Adds user's question immediately to the chat."""
    return "", history + [[user_input, None]]


def chat(history):
    user_input = history[-1][0]

    completion = buster.process_input(user_input)

    history[-1][1] = ""

    for token in completion.completor:
        history[-1][1] += token

        yield history, completion


block = gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>Buster 🤖: A Question-Answering Bot for your documentation</center></h3>")

    chatbot = gr.Chatbot()

    with gr.Row():
        question = gr.Textbox(
            label="What's your question?",
            placeholder="Ask a question to AI stackoverflow here...",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    examples = gr.Examples(
        examples=[
            "How can I perform backpropagation?",
            "How do I deal with noisy data?",
            "How do I deal with noisy data in 2 words?",
        ],
        inputs=question,
    )

    gr.Markdown("This application uses GPT to search the docs for relevant info and answer questions.")

    gr.HTML("️<center> Created with ❤️ by @jerpint and @hadrienbertrand")

    response = gr.State()

    submit.click(user, [question, chatbot], [question, chatbot], queue=False).then(
        chat, inputs=[chatbot], outputs=[chatbot, response]
    ).then(add_sources, inputs=[chatbot, response], outputs=[chatbot])
    question.submit(user, [question, chatbot], [question, chatbot], queue=False).then(
        chat, inputs=[chatbot], outputs=[chatbot, response]
    ).then(add_sources, inputs=[chatbot, response], outputs=[chatbot])


block.queue(concurrency_count=16)
block.launch(debug=True, share=False)
