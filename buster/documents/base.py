from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class DocumentsManager(ABC):
    @abstractmethod
    def add(self, source: str, df: pd.DataFrame):
        ...
