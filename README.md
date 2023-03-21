---
title: Buster
emoji: 🤖
colorFrom: red
colorTo: blue
sdk: gradio
app_file: buster/apps/gradio_app.py
python_version: 3.10.8
pinned: false
---

# Buster, the QA documentation chatbot!

Buster is a question-answering chatbot that can be tuned to any source of documentations.

# Demo

You can try out our [live demo here](https://huggingface.co/spaces/jerpint/buster), where it will answer questions about a bunch of libraries we've already scraped, including [🤗 Transformers](https://huggingface.co/docs/transformers/index).


# Quickstart

Here is a quick guide to help you deploy buster on your own dataset!

First step, install buster locally. Note that buster requires python>=3.10.

```
git clone https://github.com/jerpint/buster.git
pip install -e .
```

Then, go to the examples folder. We've attached a sample `stackoverflow.csv` file to help you get started. You will convert the .csv to a `documents.db` file.

```
buster_csv_parser stackoverflow.csv --output-filepath documents.db
```

This will generate the embeddings and save them locally. Finally, run

```
gradio gradio_app.py
```

This will launch the gradio app locally, which you should be able to view on [localhost]( http://127.0.0.1:7860)


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

- [Livestream notebook](https://colab.research.google.com/drive/1CosxSNod48KrkyBn5_vkeleb7u0CrBa6)
