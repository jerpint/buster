import os

from slack_bolt import App

from buster.chatbot import Chatbot, ChatbotConfig

UNKNOWN_PROMPT = """This doesn't seem to be related to cluster usage. I am not sure how to answer."""
MILA_CLUSTER_CHANNEL = "C04LR4H9KQA"
ORION_CHANNEL = "C04LYHGUYB0"

buster_cfg = ChatbotConfig(
    documents_csv="buster/data/document_embeddings.csv",
    unknown_prompt=UNKNOWN_PROMPT,
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
    documents_csv="buster/data/document_embeddings_orion.csv",
    unknown_prompt=UNKNOWN_PROMPT,
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

    This doesn't seem to be related to cluster usage.

    Now answer the following question:
    """,
)
orion_chatbot = Chatbot(orion_cfg)

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
                    {"type": "section", "text": {"type": "mrkdwn", "text": "*Welcome to your _App's Home_* :tada:"}},
                    {"type": "divider"},
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "This button won't do much for now but you can set up a listener for it using the `actions()` method and passing its unique `action_id`. See an example in the `examples` folder within your Bolt app.",
                        },
                    },
                    {
                        "type": "actions",
                        "elements": [{"type": "button", "text": {"type": "plain_text", "text": "Click me!"}}],
                    },
                ],
            },
        )

    except Exception as e:
        logger.error(f"Error publishing home tab: {e}")


# Start your app
if __name__ == "__main__":
    app.start(port=int(os.environ.get("PORT", 3000)))
