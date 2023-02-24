import math
import os
from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from itertools import takewhile, zip_longest
from typing import Iterator

import bs4
import pandas as pd
from bs4 import BeautifulSoup


@dataclass
class Section:
    url: str
    name: str
    nodes: InitVar[list[bs4.element.NavigableString]]
    text: str = field(init=False)

    def __post_init__(self, nodes: list[bs4.element.NavigableString]):
        section = []
        for node in nodes:
            if node.name == "table":
                node_text = pd.read_html(node.prettify())[0].to_markdown(index=False, tablefmt="github")
            elif node.name == "script":
                continue
            else:
                node_text = node.text
            section.append(node_text)
        self.text = "".join(section).strip()

    def __len__(self) -> int:
        return len(self.text)

    @classmethod
    def from_text(cls, text: str, url: str, name: str) -> "Section":
        """Alternate constructor, without parsing."""
        section = cls.__new__(cls)  # Allocate memory, does not call __init__
        # Does the init here.
        section.text = text
        section.url = url
        section.name = name

        return section

    def get_chunks(self, min_length: int, max_length: int) -> Iterator["Section"]:
        """Split a section into chunks."""
        if len(self) > max_length:
            # Get the number of chunk, by dividing and rounding up.
            # Then, split the section into equal lenght chunks.
            # This could results in chunks below the minimum length,
            # and will truncate the end of the section.
            n_chunks = (len(self) + max_length - 1) // max_length
            length = len(self) // n_chunks
            for chunk in range(n_chunks):
                start = chunk * length
                yield Section.from_text(self.text[start : start + length], self.url, self.name)
        elif len(self) > min_length:
            yield self
        return


@dataclass
class Parser(ABC):
    soup: BeautifulSoup
    base_url: str
    filename: str
    min_section_length: int = 100
    max_section_length: int = 2000

    @abstractmethod
    def build_url(self, suffix: str) -> str:
        ...

    @abstractmethod
    def find_sections(self) -> Iterator[Section]:
        ...

    def parse(self) -> list[Section]:
        """Parse the documents into sections, respecting the lenght constraints."""
        sections = []
        for section in self.find_sections():
            sections.extend(section.get_chunks(self.min_section_length, self.max_section_length))
        return sections


class SphinxParser(Parser):
    def find_sections(self) -> Iterator[Section]:
        for section in self.soup.find_all("a", href=True, class_="headerlink"):
            container = section.parent.parent
            section_href = container.find_all("a", href=True, class_="headerlink")

            url = self.build_url(section["href"].strip().replace("\n", ""))
            name = section.parent.text

            # If sections has subsections, keep only the part before the first subsection
            if len(section_href) > 1 and container.section is not None:
                siblings = list(container.section.previous_siblings)[::-1]
                section = Section(url, name, siblings)
            else:
                section = Section(url, name, container.children)
            yield section
        return

    def build_url(self, suffix: str) -> str:
        return self.base_url + self.filename + suffix


class HuggingfaceParser(Parser):
    def find_sections(self) -> Iterator[Section]:
        sections = self.soup.find_all(["h1", "h2", "h3"], class_="relative group")
        for section, next_section in zip_longest(sections, sections[1:]):
            href = section.find("a", href=True, class_="header-link")
            nodes = list(takewhile(lambda sibling: sibling != next_section, section.find_next_siblings()))

            url = self.build_url(href["href"].strip().replace("\n", ""))
            name = section.text.strip().replace("\n", "")
            yield Section(url, name, nodes)

        return

    def build_url(self, suffix: str) -> str:
        # The splitext is to remove the .html extension
        return self.base_url + os.path.splitext(self.filename)[0] + suffix
