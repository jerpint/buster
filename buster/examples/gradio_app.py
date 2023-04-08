import pandas as pd
import cfg
import gradio as gr

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


def chat(question, history):
    history = history or []
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
        gr.Markdown("<h3><center>Buster 🤖: A Question-Answering Bot for your documentation</center></h3>")

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Ask a question to AI stackoverflow here...",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    examples = gr.Examples(
        examples=[
            "How can I perform backpropagation?",
            "How do I deal with noisy data?",
        ],
        inputs=message,
    )

    gr.Markdown("This application uses GPT to search the docs for relevant info and answer questions.")

    gr.HTML("️<center> Created with ❤️ by @jerpint and @hadrienbertrand")

    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[message, state], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, state], outputs=[chatbot, state])


block.launch(debug=True, share=False)
