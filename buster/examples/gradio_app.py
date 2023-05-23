import cfg
import gradio as gr
import pandas as pd

from buster.busterbot import Buster
from buster.completers.base import ChatGPTCompleter, Completer
from buster.formatters.documents import DocumentsFormatter
from buster.formatters.prompts import PromptFormatter
from buster.retriever import Retriever, SQLiteRetriever
from buster.tokenizers import GPTTokenizer
from buster.validators.base import Validator

# initialize buster with the config in cfg.py (adapt to your needs) ...
buster_cfg = cfg.buster_cfg
retriever: Retriever = SQLiteRetriever(**buster_cfg.retriever_cfg)
tokenizer = GPTTokenizer(**buster_cfg.tokenizer_cfg)
completer: Completer = ChatGPTCompleter(
    documents_formatter=DocumentsFormatter(tokenizer=tokenizer, **buster_cfg.documents_formatter_cfg),
    prompt_formatter=PromptFormatter(tokenizer=tokenizer, **buster_cfg.prompt_formatter_cfg),
    **buster_cfg.completion_cfg,
)
validator: Validator = Validator(**buster_cfg.validator_cfg)
buster: Buster = Buster(retriever=retriever, completer=completer, validator=validator)


def format_sources(matched_documents: pd.DataFrame) -> str:
    if len(matched_documents) == 0:
        return ""

    documents_answer_template: str = (
        """📝 Here are the sources I used to answer your question:<br>""" """{documents}<br><br>""" """{footnote}"""
    )
    document_template: str = """[🔗 {document.title}]({document.url}), relevance: {document.similarity_to_answer:2.1f} %"""

    matched_documents.similarity = matched_documents.similarity * 100
    documents = "<br>".join([document_template.format(document=document) for _, document in matched_documents.iterrows()])
    footnote: str = "I'm a bot 🤖 and not always perfect."

    return documents_answer_template.format(documents=documents, footnote=footnote)


def add_sources(history, completion):
    completion = buster.postprocess_completion(completion)

    if completion.answer_relevant:
        formatted_sources = format_sources(completion.matched_documents)
        history.append([None, formatted_sources])

    return history


def user(user_input, history):
    """Adds user's question immediately to the chat."""
    return "", history + [[user_input, None]]


def chat(history):
    user_input = history[-1][0]

    completion = buster.process_input(user_input, source="stackoverflow")

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
