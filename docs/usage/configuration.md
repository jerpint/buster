# Configuration of Components

Buster's internal configuration is controlled via the `BusterConfig` object.
It is meant to set all of the different parameters for the different components in one place.

Here is a typical setup:

```python
from buster.busterbot import BusterConfig

buster_cfg = BusterConfig(
    retriever_cfg={
        "path": "deeplake_store",
        "top_k": 3,
        "thresh": 0.7,
        "max_tokens": 2000,
        "embedding_model": "text-embedding-ada-002",
    },
    validator_cfg={
        "unknown_response_templates": [
            "I'm sorry, but I am an AI language model trained to assist with questions related to AI. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with?",
        ],
        "unknown_threshold": 0.85,
        "embedding_model": "text-embedding-ada-002",
        "use_reranking": True,
        "invalid_question_response": "This question does not seem relevant to my current knowledge.",
        "check_question_prompt": """You are an chatbot answering questions on artificial intelligence.

A user will submit a question. Respond 'true' if it is valid, respond 'false' if it is invalid.""",
        "completion_kwargs": {
            "model": "gpt-3.5-turbo",
            "stream": False,
            "temperature": 0,
        },
    },
    documents_answerer_cfg={
        "no_documents_message": "No documents are available for this question.",
    },
    completion_cfg={
        "completion_kwargs": {
            "model": "gpt-3.5-turbo",
            "stream": False,
            "temperature": 0,
        },
    },
    tokenizer_cfg={
        "model_name": "gpt-3.5-turbo",
    },
    documents_formatter_cfg={
        "max_tokens": 3500,
        "columns": ["content", "title", "source"],
    },
    prompt_formatter_cfg={
        "max_tokens": 3500,
        "text_before_docs": (
            "You are a chatbot assistant answering technical questions about artificial intelligence (AI)."
            "You can only respond to a question if the content necessary to answer the question is contained in the following provided documentation. "
            "If the answer is in the documentation, summarize it in a helpful way to the user. "
        ),
        "text_after_docs": (
            "REMEMBER:\n"
            "You are a chatbot assistant answering technical questions about artificial intelligence (AI)."
            "Here are the rules you must follow:\n"
            "1) You must only respond with information contained in the documentation above. Say you do not know if the information is not provided.\n"
            "2) Make sure to format your answers in Markdown format, including code block and snippets.\n"
            "Now answer the following question:\n"
        ),
    },
)
```

This `BusterConfig` can then be passed to initialize Buster and all of its components:

```python
from buster.busterbot import Buster, BusterConfig
from buster.completers import ChatGPTCompleter, DocumentAnswerer
from buster.formatters.documents import DocumentsFormatterJSON
from buster.formatters.prompts import PromptFormatter
from buster.retriever import DeepLakeRetriever, Retriever
from buster.tokenizers import GPTTokenizer
from buster.validators import QuestionAnswerValidator, Validator

def setup_buster(buster_cfg: BusterConfig):
    """initialize buster with a buster_cfg class"""
    retriever: Retriever = DeepLakeRetriever(**buster_cfg.retriever_cfg)
    tokenizer = GPTTokenizer(**buster_cfg.tokenizer_cfg)
    document_answerer: DocumentAnswerer = DocumentAnswerer(
        completer=ChatGPTCompleter(**buster_cfg.completion_cfg),
        documents_formatter=DocumentsFormatterJSON(tokenizer=tokenizer, **buster_cfg.documents_formatter_cfg),
        prompt_formatter=PromptFormatter(tokenizer=tokenizer, **buster_cfg.prompt_formatter_cfg),
        **buster_cfg.documents_answerer_cfg,
    )
    validator: Validator = QuestionAnswerValidator(**buster_cfg.validator_cfg)
    buster: Buster = Buster(retriever=retriever, document_answerer=document_answerer, validator=validator)
    return buster

buster = setup_buster(buster_cfg)

completion = buster.process_input("What is backpropagation?")
print(completion)
```

 uses a config file to setup most of the app.