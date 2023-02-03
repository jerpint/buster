import os

from slack_bolt import App

from buster.chatbot import Chatbot, ChatbotConfig

MILA_CLUSTER_CHANNEL = "C04LR4H9KQA"
ORION_CHANNEL = "C04LYHGUYB0"
PYTORCH_CHANNEL = "C04MEK6N882"
HF_TRANSFORMERS_CHANNEL = "C04NJNCJWHE"

buster_cfg = ChatbotConfig(
    documents_file="buster/data/document_embeddings.csv",
    unknown_prompt="This doesn't seem to be related to cluster usage.",
    embedding_model="text-embedding-ada-002",
    top_k=3,
    thresh=0.7,
    max_chars=3000,
    completion_kwargs={
        "engine": "text-davinci-003",
        "max_tokens": 200,
    },
    separator="\n",
    link_format="slack",
    text_after_response="""I'm a bot 🤖 and not always perfect.
    For more info, view the full documentation here (https://docs.mila.quebec/) or contact support@mila.quebec
    """,
    text_before_prompt="""
    You are a slack chatbot assistant answering technical questions about a cluster.
    Make sure to format your answers in Markdown format, including code block and snippets.
    Do not include any links to urls or hyperlinks in your answers.

    If you do not know the answer to a question, or if it is completely irrelevant to cluster usage, simply reply with:

    'This doesn't seem to be related to cluster usage.'

    For example:

    What is the meaning of life on the cluster?

    This doesn't seem to be related to cluster usage.

    Now answer the following question:
    """,
)
buster_chatbot = Chatbot(buster_cfg)

orion_cfg = ChatbotConfig(
    documents_file="buster/data/document_embeddings_orion.csv",
    unknown_prompt="This doesn't seem to be related to the orion library. I am not sure how to answer.",
    embedding_model="text-embedding-ada-002",
    top_k=3,
    thresh=0.7,
    max_chars=3000,
    completion_kwargs={
        "engine": "text-davinci-003",
        "max_tokens": 200,
    },
    separator="\n",
    link_format="slack",
    text_after_response="I'm a bot 🤖 and not always perfect.",
    text_before_prompt="""You are a slack chatbot assistant answering technical questions about orion, a hyperparameter optimization library written in python.
    Make sure to format your answers in Markdown format, including code block and snippets.
    Do not include any links to urls or hyperlinks in your answers.

    If you do not know the answer to a question, or if it is completely irrelevant to the library usage, simply reply with:

    'This doesn't seem to be related to the orion library.'

    For example:

    What is the meaning of life for orion?

    This doesn't seem to be related to the orion library.

    Now answer the following question:
    """,
)
orion_chatbot = Chatbot(orion_cfg)

pytorch_cfg = ChatbotConfig(
    documents_file="buster/data/document_embeddings_pytorch.tar.gz",
    unknown_prompt="This doesn't seem to be related to the pytorch library. I am not sure how to answer.",
    embedding_model="text-embedding-ada-002",
    top_k=3,
    thresh=0.7,
    max_chars=3000,
    completion_kwargs={
        "engine": "text-davinci-003",
        "max_tokens": 500,
    },
    separator="\n",
    link_format="slack",
    text_after_response="I'm a bot 🤖 and not always perfect.",
    text_before_prompt="""You are a slack chatbot assistant answering technical questions about pytorch, a library to train neural networks written in python.
    Make sure to format your answers in Markdown format, including code block and snippets.
    Do not include any links to urls or hyperlinks in your answers.

    If you do not know the answer to a question, or if it is completely irrelevant to the library usage, simply reply with:

    'This doesn't seem to be related to the pytorch library.'

    For example:

    What is the meaning of life for pytorch?

    This doesn't seem to be related to the pytorch library.

    Now answer the following question:
    """,
)
pytorch_chatbot = Chatbot(pytorch_cfg)

hf_transformers_cfg = ChatbotConfig(
    documents_file="buster/data/document_embeddings_hf_transformers.tar.gz",
    unknown_prompt="This doesn't seem to be related to the huggingface library. I am not sure how to answer.",
    embedding_model="text-embedding-ada-002",
    top_k=3,
    thresh=0.7,
    max_chars=3000,
    completion_kwargs={
        "engine": "text-davinci-003",
        "max_tokens": 500,
    },
    separator="\n",
    link_format="slack",
    text_after_response="I'm a bot 🤖 and not always perfect.",
    text_before_prompt="""You are a slack chatbot assistant answering technical questions about huggingface transformers, a library to train transformers in python.
    Make sure to format your answers in Markdown format, including code block and snippets.
    Do not include any links to urls or hyperlinks in your answers.

    If you do not know the answer to a question, or if it is completely irrelevant to the library usage, simply reply with:

    'This doesn't seem to be related to the huggingface library.'

    For example:

    What is the meaning of life for huggingface?

    This doesn't seem to be related to the huggingface library.

    Now answer the following question:
    """,
)
hf_transformers_chatbot = Chatbot(hf_transformers_cfg)

app = App(token=os.environ.get("SLACK_BOT_TOKEN"), signing_secret=os.environ.get("SLACK_SIGNING_SECRET"))


@app.event("app_mention")
def respond_to_question(event, say):
    print(event)

    # user's text
    text = event["text"]
    channel = event["channel"]

    if channel == MILA_CLUSTER_CHANNEL:
        print("*******using BUSTER********")
        answer = buster_chatbot.process_input(text)
    elif channel == ORION_CHANNEL:
        print("*******using ORION********")
        answer = orion_chatbot.process_input(text)
    elif channel == PYTORCH_CHANNEL:
        print("*******using PYTORCH********")
        answer = pytorch_chatbot.process_input(text)
    elif channel == HF_TRANSFORMERS_CHANNEL:
        print("*******using HF TRANSFORMERS********")
        answer = hf_transformers_chatbot.process_input(text)
    else:
        print(f"invalid channel: {channel}")
        answer = "I was not yet implemented to support this channel."

    # responds to the message in the thread
    thread_ts = event["event_ts"]

    say(text=answer, thread_ts=thread_ts)


@app.event("app_home_opened")
def update_home_tab(client, event, logger):
    try:
        # views.publish is the method that your app uses to push a view to the Home tab
        client.views_publish(
            # the user that opened your app's app home
            user_id=event["user"],
            # the view object that appears in the app home
            view={
                "type": "home",
                "callback_id": "home_view",
                # body of the view
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": "*Hello, I'm _BusterBot_* :tada:"}},
                    {"type": "divider"},
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": (
                                "I am a chatbot 🤖 designed to answer questions related to technical documentation.\n\n"
                                "I use OpenAI's GPT models to target which relevant sections of documentation are relevant and respond with.\n"
                                "I am open-sourced, and my code is available on github: https://github.com/jerpint/buster\n\n"
                                "For more information, contact either Jeremy or Hadrien from the AMLRT team.\n"
                            ),
                        },
                    },
                    # {
                    #     "type": "actions",
                    #     "elements": [{"type": "button", "text": {"type": "plain_text", "text": "Click me!"}}],
                    # },
                ],
            },
        )

    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")


# Start your app
if __name__ == "__main__":
    app.start(port=int(os.environ.get("PORT", 3000)))
