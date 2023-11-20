from buster.busterbot import Buster, BusterConfig
from buster.completers import ChatGPTCompleter, DocumentAnswerer
from buster.formatters.documents import DocumentsFormatterJSON
from buster.formatters.prompts import PromptFormatter
from buster.llm_utils import get_openai_embedding_constructor
from buster.retriever import DeepLakeRetriever, Retriever
from buster.tokenizers import GPTTokenizer
from buster.validators import Validator

# kwargs to pass to OpenAI client
client_kwargs = {
    "timeout": 20,
    "max_retries": 3,
}

embedding_fn = get_openai_embedding_constructor(client_kwargs=client_kwargs)

buster_cfg = BusterConfig(
    validator_cfg={
        "question_validator_cfg": {
            "invalid_question_response": "This question does not seem relevant to my current knowledge.",
            "completion_kwargs": {
                "model": "gpt-3.5-turbo",
                "stream": False,
                "temperature": 0,
            },
            "client_kwargs": client_kwargs,
            "check_question_prompt": """You are a chatbot answering questions on artificial intelligence.
Your job is to determine wether or not a question is valid, and should be answered.
More general questions are not considered valid, even if you might know the response.
A user will submit a question. Respond 'true' if it is valid, respond 'false' if it is invalid.

For example:

Q: What is backpropagation?
true

Q: What is the meaning of life?
false

A user will submit a question. Respond 'true' if it is valid, respond 'false' if it is invalid.""",
        },
        "answer_validator_cfg": {
            "unknown_response_templates": [
                "I'm sorry, but I am an AI language model trained to assist with questions related to AI. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with?",
            ],
            "unknown_threshold": 0.85,
            "embedding_fn": embedding_fn,
        },
        "documents_validator_cfg": {
            "completion_kwargs": {
                "model": "gpt-3.5-turbo",
                "stream": False,
                "temperature": 0,
            },
            "client_kwargs": client_kwargs,
        },
        "use_reranking": True,
        "validate_documents": False,
    },
    retriever_cfg={
        "path": "deeplake_store",
        "top_k": 3,
        "thresh": 0.7,
        "embedding_fn": embedding_fn,
    },
    documents_answerer_cfg={
        "no_documents_message": "No documents are available for this question.",
    },
    completion_cfg={
        "completion_kwargs": {
            "model": "gpt-3.5-turbo",
            "stream": True,
            "temperature": 0,
        },
        "client_kwargs": client_kwargs,
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
            "If it isn't, simply reply that you cannot answer the question. "
            "Do not refer to the documentation directly, but use the instructions provided within it to answer questions. "
            "Here is the documentation: "
        ),
        "text_after_docs": (
            "REMEMBER:\n"
            "You are a chatbot assistant answering technical questions about artificial intelligence (AI)."
            "Here are the rules you must follow:\n"
            "1) You must only respond with information contained in the documentation above. Say you do not know if the information is not provided.\n"
            "2) Make sure to format your answers in Markdown format, including code block and snippets.\n"
            "3) Do not reference any links, urls or hyperlinks in your answers.\n"
            "4) If you do not know the answer to a question, or if it is completely irrelevant to the library usage, simply reply with:\n"
            "5) Do not refer to the documentation directly, but use the instructions provided within it to answer questions. "
            "'I'm sorry, but I am an AI language model trained to assist with questions related to AI. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with?'"
            "For example:\n"
            "What is the meaning of life for an qa bot?\n"
            "I'm sorry, but I am an AI language model trained to assist with questions related to AI. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with?"
            "Now answer the following question:\n"
        ),
    },
)


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
    validator: Validator = Validator(**buster_cfg.validator_cfg)
    buster: Buster = Buster(retriever=retriever, document_answerer=document_answerer, validator=validator)
    return buster
