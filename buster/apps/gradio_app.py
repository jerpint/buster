import os
import pathlib

import gradio as gr

import git


from buster.buster import Buster, BusterConfig


def get_git_working_dir():
    path = pathlib.Path(__file__).resolve()
    git_repo = git.Repo(path, search_parent_directories=True)
    return git_repo.working_dir


working_dir = get_git_working_dir()
buster_cfg = BusterConfig(
    documents_file=os.path.join(os.path.join(working_dir, "buster/data/document_embeddings_huggingface.tar.gz")),
    unknown_prompt="I'm sorry, but I am an AI language model trained to assist with questions related to the huggingface transformers library. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with?",
    embedding_model="text-embedding-ada-002",
    top_k=3,
    thresh=0.7,
    max_words=3000,
    completer_cfg={
        "name": "ChatGPT",
        "text_before_documents": (
            "You are a chatbot assistant answering technical questions about huggingface transformers, a library to train transformers in python. "
            "You can only respond to a question if the content necessary to answer the question is contained in the following provided documentation. "
            "If it isn't, simply reply that you cannot answer the question. "
            "Here is the documentation: "
            "<BEGIN_DOCUMENTATION> "
        ),
        "text_before_prompt": (
            "<\END_DOCUMENTATION>\n"
            "REMINDER:\n"
            "You are a chatbot assistant answering technical questions about huggingface transformers, a library to train transformers in python. "
            "Here are the rules you must follow:\n"
            "1) You must only respond with information contained in the documentation above. Say you do not know if the information is not provided.\n"
            "2) Make sure to format your answers in Markdown format, including code block and snippets.\n"
            "3) Do not include any links to urls or hyperlinks in your answers.\n"
            "4) If you do not know the answer to a question, or if it is completely irrelevant to the library usage, simply reply with:\n"
            "'I'm sorry, but I am an AI language model trained to assist with questions related to the huggingface transformers library. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with?'"
            "For example:\n"
            "What is the meaning of life for huggingface?\n"
            "I'm sorry, but I am an AI language model trained to assist with questions related to the huggingface transformers library. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with?"
            "Now answer the following question:\n"
        ),
        "completion_kwargs": {
            "model": "gpt-3.5-turbo",
        },
    },
    response_format="gradio",
)
buster = Buster(buster_cfg)


def chat(question, history):
    history = history or []

    answer = buster.process_input(question)

    # formatting hack for code blocks to render properly every time
    answer = answer.replace("```", "\n```\n")

    history.append((question, answer))
    return history, history


block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>Buster 🤖: A Question-Answering Bot for Huggingface 🤗 Transformers </center></h3>")

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="What kind of model should I use for sentiment analysis?",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
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


block.launch(debug=True)
