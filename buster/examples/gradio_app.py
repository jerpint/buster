import cfg
import gradio as gr

from buster.busterbot import Buster
from buster.retriever import Retriever
from buster.utils import get_retriever_from_extension

# initialize buster with the config in config.py (adapt to your needs) ...
documents: Retriever = get_retriever_from_extension(cfg.documents_filepath)(cfg.documents_filepath)
buster: Buster = Buster(cfg=cfg.buster_cfg, documents=documents)


def chat(question, history):
    history = history or []
    answer = buster.process_input(question)

    # formatting hack for code blocks to render properly every time
    answer = answer.replace("```", "\n```\n")

    history.append((question, answer))
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
