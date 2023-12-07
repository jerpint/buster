from dataclasses import dataclass
from typing import Optional


@dataclass
class UserInputs:
    """A class that represents user inputs.

    Attributes:
        original_input: The original user input.
        reformulated_input: The reformulated user input (optional).
    """

    original_input: str
    reformulated_input: Optional[str] = None

    @property
    def current_input(self):
        """Returns the current user input.

        If the reformulated input is not None, it returns the reformulated input.
        Otherwise, it returns the original input.

        Returns:
            The current user input.
        """
        return self.reformulated_input if self.reformulated_input is not None else self.original_input
