import os
import pathlib

import gradio as gr

from buster.buster import Buster, BusterConfig
from buster.documents.utils import get_documents_manager_from_extension
from buster.apps.bot_configs import get_config

DATA_DIR = pathlib.Path(__file__).parent.parent.resolve() / "data"  # points to ../data/

documents_filepath=os.path.join(DATA_DIR, "documents.db")
documents = get_documents_manager_from_extension(documents_filepath)(documents_filepath)

buster = Buster(documents=documents)

def chat(question, history):
    history = history or []


    answer = buster.process_input(question)

    # formatting hack for code blocks to render properly every time
    answer = answer.replace("```", "\n```\n")

    history.append((question, answer))
    return history, history


def update_config(bot_source: str):
    buster.cfg = get_config(source=bot_source)

block = gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>Buster 🤖: A Question-Answering Bot for Huggingface 🤗 Transformers </center></h3>")

    doc_source = gr.Dropdown(
        choices=["huggingface", "pytorch"] , value="huggingface", interactive=True, multiselect=False, label="Source of Documentation", info="The source of documentation to select from"
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
        examples=[
            "What kind of models should I use for images and text?",
            "When should I finetune a model vs. training it form scratch?",
            "How can I deploy my trained huggingface model?",
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

    submit.click(chat, inputs=[message, state], outputs=[chatbot, state])
    message.submit(chat, inputs=[message, state], outputs=[chatbot, state])
    doc_source.change(update_config, inputs=[doc_source])


block.launch(debug=True)
