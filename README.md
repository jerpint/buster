# Buster, the QA documentation chatbot!

<div align="center">

[![GitHub](https://img.shields.io/github/license/jerpint/buster)](https://github.com/jerpint/buster)
[![PyPI](https://img.shields.io/pypi/v/buster-doctalk?logo=pypi)](https://pypi.org/project/buster-doctalk)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Buster%20Demo-blue)](https://huggingface.co/spaces/jerpint/buster)

</div>

Buster is a question-answering chatbot that can be tuned to any source of documentations.

# Demo

In order to view the full abilities of Buster, you can play with our [live demo here](https://huggingface.co/spaces/jerpint/buster).
We scraped the documentation of [huggingface 🤗 Transformers](https://huggingface.co/docs/transformers/index) and instructed Buster to answer questions related to its usage.

# Quickstart

This section is meant to help you install and run local version of Buster.
First step, install buster:

**Note**: Buster requires python>=3.10

```bash
pip install buster-doctalk
```

Then, go to the examples folder and launch the app.
We've included small sample data off stackoverflow-ai questions that you can test your setup with to try app:

```bash
cd buster/buster/examples
gradio gradio_app.py
```

This will launch the gradio app locally.


**NOTE**: The demo uses chatGPT to generate text and compute embeddings, make sure to set a valid openai API key:
```bash
export OPENAI_API_KEY=sk-...
```

# Generating your own embeddings

Once your local version of Buster is up and running, the next step is for you to be able to import your own data.
We will be using the `stackoverflow.csv` file in the `buster/examples/` folder for this. This is the same data that was used to generate the demo app's embeddings.

You will first ingest the documents to be ready for buster. In this example, we use Deeplake's vector store, but you can always write your own custom `DocumentManager`:


```python
import pandas as pd
from buster.documents_manager import DeepLakeDocumentsManager

# Read the csv
df = pd.read_csv("stackoverflow.csv")

# Generate the embeddings for our documents and store them in a deeplake format
dm = DeepLakeDocumentsManager(vector_store_path="deeplake_store", overwrite=True)
dm.add(df)
```

You can also just simply run the script:

    python generate_embeddings.py --csv stackoverflow.csv


This will generate the embeddings and save them locally in the `deeplake_store`.


**NOTE**: You will need to set a valid openai key for computing embeddings:

```bash
export OPENAI_API_KEY=sk-...
```

You only need to run this operation one time.

In the .csv, we expect columns ["title", "url", "content", "source"] for each row of the csv:

* title: this will be the title of the url to display
* url: the link that clicking the title will redirect to
* source: where the content was originally sourced from (e.g. wikipedia, stackoverflow, etc.)
* content: plaintext of the documents to be embedded. It is your responsibility to chunk your documents appropriately. For better results, we recommend chunks of 400-600 words.

# Additional Configurations

Properly prompting models as well as playing around with various model parameters can lead to different results. We use a `BusterConfig` object to keep track of the various Buster configurations. In the `buster/examples/` folder, the config is stored inside `cfg.py`. Modify this config to update parameters, prompts, etc.

# How does Buster work?

First, we parsed the documentation into snippets. For each snippet, we obtain an embedding by using the [OpenAI API](https://beta.openai.com/docs/guides/embeddings/what-are-embeddings).

Then, when a user asks a question, we compute its embedding, and find the snippets from the doc with the highest cosine similarity to the question.

Finally, we craft the prompt:
- The most relevant snippets from the doc.
- The engineering prompt.
- The user's question.

We send the prompt to the [OpenAI API](https://beta.openai.com/docs/api-reference/completions), and display the answer to the user!

### Currently available models

- For embeddings: "text-embedding-ada-002"
- For completion: We support both "gpt-3.5-turbo" and "gpt-4"

### Livestream

For more information, you can watch the livestream where explain how buster works in detail!

- [Livestream recording](https://youtu.be/LB5g-AhfPG8)