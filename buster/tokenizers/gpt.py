import tiktoken

from buster.tokenizers import Tokenizer


class GPTTokenizer(Tokenizer):
    """Tokenizer from openai, supports most GPT models."""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.encoder = tiktoken.encoding_for_model(model_name=model_name)

    def encode(self, string: str):
        return self.encoder.encode(string)

    def decode(self, encoded: list[int]):
        return self.encoder.decode(encoded)
