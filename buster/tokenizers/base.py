from abc import ABC, abstractmethod
from typing import Union


class Tokenizer(ABC):
    """Abstract base class for a tokenizer."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def encode(self, string: str) -> list[int]:
        ...

    @abstractmethod
    def decode(self, encoded: list[int]) -> str:
        ...

    def num_tokens(self, string: str, return_encoded: bool = False) -> Union[int, tuple[int, list[int]]]:
        encoded = self.encode(string)
        if return_encoded:
            return len(encoded), encoded
        return len(encoded)
