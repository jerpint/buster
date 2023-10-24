# Installation

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