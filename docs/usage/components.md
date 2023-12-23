# Buster Components

Buster is built around components that can be customized and extended.

For example, to do chat completion, we must use a `Completer` component.
While we've implemented some completers like `ChatGPT`, adding more completers is possible by inheriting from the `Completer` base class.

Currently, buster implements the following components:

* `Completer`: The language model responsible for generating a response
* `Retriever`: Responsible for fetching the documents associated to a user's input
* `DocumentsFormatter`: Responsible for taking the various documents and formatting them in different ways. We support formatting documents into json-like objects and html-like objects.
* `PromptFormatter`: Responsible for combining the formatted documents with the prompts for the LLM
* `Validator`: Responsible for validating user inputs and/or model outputs. This can be implemented via checks of the questions and answer before and after completions occur.
* `Tokenizer`: Used to monitor the length of prompts and completions. It is generally assumed that the `Tokenizer` is associated to that of the `Completer`.


Additional components are also available for managing documents:
* `DocumentManager`: Manager allowing to generate and store embeddings (should be used in conjunction with `Retriever` components)