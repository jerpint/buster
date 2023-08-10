# Buster, the QA documentation chatbot!

Buster is a question-answering chatbot that can be tuned to any source of documentations.

# Demo

You can try out our [live demo here](https://huggingface.co/spaces/jerpint/buster), where it will answer questions about a bunch of libraries we've already scraped, including [🤗 Transformers](https://huggingface.co/docs/transformers/index).


# Quickstart

Here is a quick guide to help you deploy buster on your own dataset!

We will look at deploying a simple app locally.
First step, install buster. Note that buster requires python>=3.10.

```
git clone https://github.com/jerpint/buster.git
pip install .
```

Then, go to the examples folder:

    cd buster/buster/examples

We've attached a sample `stackoverflow.csv` file to help you get started.

You will first ingest the documents to be ready for buster. In this example, we use Deeplake's vectore store, but you can always write your own custom `DocumentManager`:


    import pandas as pd
    from buster.documents_manager import DeepLakeDocumentsManager

    # Read the csv
    df = pd.read_csv("stackoverflow.csv")

    # Generate the embeddings for our documents and store them in a deeplake format
    dm = DeepLakeDocumentsManager(vector_store_path="deeplake_store", overwrite=True)
    dm.add(df)

You can also just simply run the script:

    python generate_embeddings.py


This will generate the embeddings and save them locally in the `deeplake_store` folder.
Note: You only need to run this operation one time.

Now, you can launch your gradio app:

    gradio gradio_app.py

This will launch the gradio app locally, which you should be able to view on [localhost]( http://127.0.0.1:7860)

In the .csv, we expect columns ["title", "url", "content", "source"] for each row of the csv:

* title: this will be the title of the url to display
* url: the actual link that will be shown to the user
* source: where the content was originally sourced from (e.g. wikipedia, medium, etc.)
* content: plaintext of the document to be embedded. Note that we do not do any chunking (yet). It is your responsibility to ensure each document is of an appropriate context length.

## How does Buster work?

First, we parsed the documentation into snippets. For each snippet, we obtain an embedding by using the [OpenAI API](https://beta.openai.com/docs/guides/embeddings/what-are-embeddings).

Then, when a user asks a question, we compute its embedding, and find the snippets from the doc with the highest cosine similarity to the question.

Finally, we craft the prompt:
- The most relevant snippets from the doc.
- The engineering prompt.
- The user's question.

We send the prompt to the [OpenAI API](https://beta.openai.com/docs/api-reference/completions), and display the answer to the user!

### Currently available models

- For embeddings: "text-embedding-ada-002"
- For completion: We support both "text-davinci-003" and "gpt-3.5-turbo"

### Livestream

For more information, you can watch the livestream where explain how buster works in detail!

- [Livestream recording](https://youtu.be/LB5g-AhfPG8)

- [Livestream notebook](https://colab.research.google.com/drive/1CosxSNod48KrkyBn5_vkeleb7u0CrBa6) (Note this notebook is deprecated and not maintained anymore)
