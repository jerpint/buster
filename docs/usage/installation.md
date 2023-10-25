# Installation

## Install with pip

This section is meant to help you install and run local version of Buster.
First step, install buster:

**Note**: Buster requires python>=3.10

```bash
pip install buster-doctalk
```

We recommend using a virtual environment (e.g. via conda) for the installation process.

## Testing your installation

To easily test your setup, we've added an example app in the `buster/examples` directory with a few documents from stackoverflow AI questions.


**NOTE**: The demo uses chatGPT to generate text and compute embeddings, make sure to set a valid openai API key:
```bash
export OPENAI_API_KEY=sk-...
```

Simply go to the examples folder and launch the app:

```bash
cd buster/buster/examples
gradio gradio_app.py
```

This will launch the gradio app locally. Navigate to your local gradio instance and you should see the chat interface:

![image](https://github.com/jerpint/buster/assets/18450628/1604a3a9-0bee-4cd2-a6ca-70e88ddc0b81)



## Install from source

If you want to contribute to buster and improve the library, we recommend installing it locally in editable mode. To do that, clone the repository then install the library:

    git clone https://github.com/jerpint/buster
    cd buster/
    pip install -e .

