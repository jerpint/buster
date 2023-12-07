import logging
import os
from typing import Iterator, Optional

import openai
from openai import OpenAI

from buster.completers import Completer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Check if an API key exists for promptlayer, if it does, use it
promptlayer_api_key = os.environ.get("PROMPTLAYER_API_KEY")
if promptlayer_api_key:
    # TODO: Check if this still works with latest openAI API...
    try:
        import promptlayer

        logger.info("Enabling prompt layer...")
        promptlayer.api_key = promptlayer_api_key

        # replace openai with the promptlayer wrapper
        openai = promptlayer.openai
    except Exception as e:
        logger.exception("Something went wrong enabling promptlayer.")


class ChatGPTCompleter(Completer):
    def __init__(self, completion_kwargs: dict, client_kwargs: Optional[dict] = None):
        """Initialize the ChatGPTCompleter with completion and client keyword arguments.

        Args:
          completion_kwargs: A dictionary of keyword arguments to be used for completions.
          client_kwargs: An optional dictionary of keyword arguments to be used for the OpenAI client.
        """
        # use default client if none passed
        self.completion_kwargs = completion_kwargs

        if client_kwargs is None:
            client_kwargs = {}

        self.client = OpenAI(**client_kwargs)

    def complete(self, prompt: str, user_input: str, completion_kwargs=None) -> (str | Iterator, bool):
        """Given a prompt and user input, returns the generated message and error flag.

        Args:
          prompt: The prompt containing the formatted documents and instructions.
          user_input: The user input to be responded to.
          completion_kwargs: An optional dictionary of keyword arguments to override the default completion kwargs.

        Returns:
          A tuple containing the completed message and a boolean indicating if an error occurred.

        Raises:
          openai.BadRequestError: If the completion request is invalid.
          openai.RateLimitError: If the OpenAI servers are overloaded.
        """
        # Uses default configuration if not overridden

        if completion_kwargs is None:
            completion_kwargs = self.completion_kwargs

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ]

        try:
            error = False
            response = self.client.chat.completions.create(messages=messages, **completion_kwargs)
        except openai.BadRequestError:
            error = True
            logger.exception("Invalid request to OpenAI API. See traceback:")
            error_message = "Something went wrong while connecting with OpenAI, try again soon!"
            return error_message, error

        except openai.RateLimitError:
            error = True
            logger.exception("RateLimit error from OpenAI. See traceback:")
            error_message = "OpenAI servers seem to be overloaded, try again later!"
            return error_message, error

        except Exception as e:
            error = True
            logger.exception("Some kind of error happened trying to generate the response. See traceback:")
            error_message = "Something went wrong with connecting with OpenAI, try again soon!"
            return error_message, error

        if completion_kwargs.get("stream") is True:
            # We are entering streaming mode, so here we're just wrapping the streamed
            # openai response to be easier to handle later
            def answer_generator():
                for chunk in response:
                    token = chunk.choices[0].delta.content

                    # Always stream a string, openAI returns None on last token
                    token = "" if token is None else token

                    yield token

            return answer_generator(), error

        else:
            full_response: str = response.choices[0].message.content
            return full_response, error
