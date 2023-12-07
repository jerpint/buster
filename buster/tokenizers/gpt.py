import tiktoken

from buster.tokenizers import Tokenizer


class GPTTokenizer(Tokenizer):
    """Tokenizer class for GPT models.

    This class implements a tokenizer for GPT models using the tiktoken library.

    Args:
        model_name (str): The name of the GPT model to be used.

    Attributes:
        encoder: The encoder object created using tiktoken.encoding_for_model().

    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.encoder = tiktoken.encoding_for_model(model_name=model_name)

    def encode(self, string: str):
        """Encodes a given string using the GPT tokenizer.

        Args:
            string (str): The string to be encoded.

        Returns:
            list[int]: The encoded representation of the string.

        """
        return self.encoder.encode(string)

    def decode(self, encoded: list[int]):
        """Decodes a list of tokens using the GPT tokenizer.

        Args:
            encoded (list[int]): The list of tokens to be decoded.

        Returns:
            str: The decoded string representation of the tokens.

        """
        return self.encoder.decode(encoded)
