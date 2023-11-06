from dataclasses import dataclass
from typing import Optional


@dataclass
class UserInputs:
    original_input: str
    reformulated_input: Optional[str] = None

    @property
    def current_input(self):
        return self.reformulated_input if self.reformulated_input is not None else self.original_input
