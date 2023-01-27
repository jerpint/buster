import os

from slack_bolt import App

from buster.chatbot import answer_question, load_embeddings

df = load_embeddings("buster/data/document_embeddings.csv")

# Initializes your app with your bot token and signing secret
app = App(token=os.environ.get("SLACK_BOT_TOKEN"), signing_secret=os.environ.get("SLACK_SIGNING_SECRET"))

# Add functionality here
# @app.event("app_home_opened") etc


@app.event("app_mention")
def respond_to_question(event, say):
    print(event)

    # user's text
    text = event["text"]
    # text = event['blocks'][0]['elements'][0]['elements'][1]['text']
    print(text)

    answer = answer_question(text, df, top_k=3, thresh=0.7, style="md")

    # responds to the message in the thread
    thread_ts = event["event_ts"]

    # user_id = event["user"]
    # text = f"Welcome to the team, <@{user_id}>! 🎉 You can introduce yourself in this channel."
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
