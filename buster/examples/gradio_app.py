import cfg
import gradio as gr
import pandas as pd

from buster.busterbot import Buster
from buster.retriever import Retriever
from buster.utils import get_retriever_from_extension

# initialize buster with the config in config.py (adapt to your needs) ...
retriever: Retriever = get_retriever_from_extension(cfg.documents_filepath)(cfg.documents_filepath)
buster: Buster = Buster(cfg=cfg.buster_cfg, retriever=retriever)


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


def add_sources(history, response):
    response_relevant = response.response_relevant

    if response_relevant:
        # add sources
        formatted_sources = format_sources(response.matched_documents)
        history.append([None, formatted_sources])

    return history


def user(user_input, history):
    """Adds user's question immediately to the chat."""
    return "", history + [[user_input, None]]


def chat(history):
    user_input = history[-1][0]

    response = buster.process_input(user_input)

    history[-1][1] = ""

    for token in response.completion.completor:
        history[-1][1] += token

        yield history, response


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
