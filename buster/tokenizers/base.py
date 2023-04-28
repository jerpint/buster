from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """Abstract base class for a tokenizer."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def encode(self, string: str) -> list[int]:
        ...

    @abstractmethod
    def decode(self, encoded: list[int]):
        ...

    def num_tokens(self, string: str, return_encoded: bool = False):
        encoded = self.encode(string)
        if return_encoded:
            return len(encoded), encoded
        return len(encoded)
