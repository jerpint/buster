from .base import Tokenizer
from .gpt import GPTTokenizer


def tokenizer_factory(tokenizer_cfg: dict) -> Tokenizer:
    model_name = tokenizer_cfg["model_name"]
    if model_name in ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"]:
        return GPTTokenizer(model_name)

    raise ValueError(f"Tokenizer not implemented for {model_name=}")


__all__ = [Tokenizer, GPTTokenizer, tokenizer_factory]
