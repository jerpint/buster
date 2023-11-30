from abc import ABC, abstractmethod
from typing import Union


class Tokenizer(ABC):
    """Abstract base class for a tokenizer.

    Args:
      model_name: The name of the tokenizer model.

    Attributes:
      model_name: The name of the tokenizer model.

    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def encode(self, string: str) -> list[int]:
        """Encodes a string into a list of integers.

        Args:
          string: The input string to be encoded.

        Returns:
          A list of integers representing the encoded string.

        """

        ...

    @abstractmethod
    def decode(self, encoded: list[int]) -> str:
        """Decodes a list of integers into a string.

        Args:
          encoded: The list of integers to be decoded.

        Returns:
          The decoded string.

        """

        ...

    def num_tokens(self, string: str, return_encoded: bool = False) -> Union[int, tuple[int, list[int]]]:
        """Returns the number of tokens in a string.

        Args:
          string: The input string.
          return_encoded: Whether or not to return the encoded string along with the number of tokens.

        Returns:
          If `return_encoded` is False, returns the number of tokens in the string.
          If `return_encoded` is True, returns a tuple containing the number of tokens and the encoded string.

        """

        encoded = self.encode(string)
        if return_encoded:
            return len(encoded), encoded
        return len(encoded)
