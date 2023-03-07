---
title: Buster
emoji: 🤖
colorFrom: red
colorTo: blue
sdk: gradio
app_file: buster/apps/gradio_app.py
python_version: 3.10
pinned: false
---

# Buster, the QA documentation chatbot!

Buster is a question-answering chatbot that can be tuned to specific documentations. You can try it [here](https://huggingface.co/spaces/jerpint/buster), where it will answer questions about [🤗 Transformers](https://huggingface.co/docs/transformers/index).


![Question: How do I load a Huggingface model?](buster/imgs/qa_web_load.png)

![Question: My code is crashing with "CUDA out of memory". What can I do to solve this?](buster/imgs/qa_web_oom.png)

## How does Buster work?

First, we parsed the documentation into snippets. For each snippet, we obtain an embedding by using the [OpenAI API](https://beta.openai.com/docs/guides/embeddings/what-are-embeddings).

Then, when a user asks a question, we compute its embedding, and find the snippets from the doc with the highest cosine similarity to the question.

Finally, we craft the prompt:
- The most relevant snippets from the doc.
- The engineering prompt.
- The user's question.

We send the prompt to the [OpenAI API](https://beta.openai.com/docs/api-reference/completions), and display the answer to the user!

### Currently used models

- For embeddings: "text-embedding-ada-002"
- For completion: We support both "text-davinci-003" and "gpt-3.5-turbo"

### Livestream

For more information, you can watch the livestream where explain how buster works in detail!

- [Livestream recording](https://youtu.be/LB5g-AhfPG8)

- [Livestream notebook](https://colab.research.google.com/drive/1CosxSNod48KrkyBn5_vkeleb7u0CrBa6)
