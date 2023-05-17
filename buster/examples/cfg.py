from buster.busterbot import BusterConfig

documents_filepath = "./documents.db"
buster_cfg = BusterConfig(
    validator_cfg={
        "unknown_prompt": "I'm sorry, but I am an AI language model trained to assist with questions related to AI. I cannot answer that question as it is not relevant to the library or its usage. Is there anything else I can assist you with?",
        "unknown_threshold": 0.85,
        "embedding_model": "text-embedding-ada-002",
    },
    retriever_cfg={
        "top_k": 3,
        "thresh": 0.7,
        "max_tokens": 2000,
        "embedding_model": "text-embedding-ada-002",
    },
    completion_cfg={
        "name": "ChatGPT",
        "completion_kwargs": {
            "model": "gpt-3.5-turbo",
            "stream": True,
            "temperature": 0,
        },
    },
    tokenizer_cfg={
        "model_name": "gpt-3.5-turbo",
    },
    documents_formatter_cfg={
        "max_tokens": 3500,
        "format_str": "{content}",
    },
    prompt_cfg={
        "max_tokens": 3500,
        "text_before_documents": (
            "You are a chatbot assistant answering technical questions about artificial intelligence (AI)."
            "You can only respond to a question if the content necessary to answer the question is contained in the following provided documentation. "
            "If the answer is in the documentation, summarize it in a helpful way to the user. "
            "If it isn't, simply reply that you cannot answer the question. "
            "Do not refer to the documentation directly, but use the instructions provided within it to answer questions. "
            "Here is the documentation: "
            "<DOCUMENTS> "
        ),
        "text_before_prompt": (
            "<\DOCUMENTS>\n"
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
    document_source="stackoverflow",
)
