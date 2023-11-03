import os
from typing import Optional, Tuple

import cfg
import gradio as gr
import pandas as pd
from cfg import setup_buster

from buster.completers import Completion
from buster.utils import extract_zip

# Check if an openai key is set as an env. variable
if os.getenv("OPENAI_API_KEY") is None:
    print("Warning: No openai key detected. You can set it with 'export OPENAI_API_KEY=sk-...'.")

# Typehint for chatbot history
ChatHistory = list[list[Optional[str], Optional[str]]]

extract_zip("deeplake_store.zip", "deeplake_store")

buster = setup_buster(cfg.buster_cfg)


def add_user_question(user_question: str, chat_history: Optional[ChatHistory] = None) -> ChatHistory:
    """Adds a user's question to the chat history.

    If no history is provided, the first element of the history will be the user conversation.
    """
    if chat_history is None:
        chat_history = []
    chat_history.append([user_question, None])
    return chat_history


def format_sources(matched_documents: pd.DataFrame) -> str:
    if len(matched_documents) == 0:
        return ""

    matched_documents.similarity_to_answer = matched_documents.similarity_to_answer * 100

    # drop duplicate pages (by title), keep highest ranking ones
    matched_documents = matched_documents.sort_values("similarity_to_answer", ascending=False).drop_duplicates(
        "title", keep="first"
    )

    documents_answer_template: str = (
        "📝 Here are the sources I used to answer your question:\n\n{documents}\n\n{footnote}"
    )
    document_template: str = "[🔗 {document.title}]({document.url}), relevance: {document.similarity_to_answer:2.1f} %"

    documents = "\n".join([document_template.format(document=document) for _, document in matched_documents.iterrows()])
    footnote: str = "I'm a bot 🤖 and not always perfect."

    return documents_answer_template.format(documents=documents, footnote=footnote)


def add_sources(history, completion):
    if completion.answer_relevant:
        formatted_sources = format_sources(completion.matched_documents)
        history.append([None, formatted_sources])

    return history


def chat(chat_history: ChatHistory) -> Tuple[ChatHistory, Completion]:
    """Answer a user's question using retrieval augmented generation."""

    # We assume that the question is the user's last interaction
    user_input = chat_history[-1][0]

    # Do retrieval + augmented generation with buster
    completion = buster.process_input(user_input)

    # Stream tokens one at a time to the user
    chat_history[-1][1] = ""
    for token in completion.answer_generator:
        chat_history[-1][1] += token

        yield chat_history, completion


demo = gr.Blocks()

with demo:
    with gr.Row():
        gr.Markdown("<h3><center>Buster 🤖: A Question-Answering Bot for your documentation</center></h3>")

    chatbot = gr.Chatbot()

    with gr.Row():
        question_textbox = gr.Textbox(
            label="What's your question?",
            placeholder="Type your question here...",
            lines=1,
        )
        send_button = gr.Button(value="Send", variant="secondary")

    examples = gr.Examples(
        examples=[
            "How can I perform backpropagation?",
            "How do I deal with noisy data?",
            "How do I deal with noisy data in 2 words?",
        ],
        inputs=question_textbox,
    )

    gr.Markdown("This application uses GPT to search the docs for relevant info and answer questions.")

    gr.HTML("️<center> Created with ❤️ by @jerpint and @hadrienbertrand")

    response = gr.State()

    # fmt: off
    gr.on(
        triggers=[send_button.click, question_textbox.submit],
        fn=add_user_question,
        inputs=[question_textbox],
        outputs=[chatbot]
    ).then(
        chat,
        inputs=[chatbot],
        outputs=[chatbot, response]
    ).then(
        add_sources,
        inputs=[chatbot, response],
        outputs=[chatbot]
    )

    # fmt: on


demo.queue()
demo.launch(debug=True, share=False)
