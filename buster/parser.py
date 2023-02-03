import math
import os

import bs4
import pandas as pd
from bs4 import BeautifulSoup


def parse_section(nodes: list[bs4.element.NavigableString]) -> str:
    section = []
    for node in nodes:
        if node.name == "table":
            node_text = pd.read_html(node.prettify())[0].to_markdown(index=False, tablefmt="github")
        elif node.name == "script":
            continue
        else:
            node_text = node.text
        section.append(node_text)
    section = "".join(section)

    return section


class Parser:
    def __init__(
        self,
        soup: BeautifulSoup,
        base_url: str,
        filename: str,
        min_section_length: int = 100,
        max_section_length: int = 2000,
    ):
        self.soup = soup
        self.base_url = base_url
        self.filename = filename
        self.min_section_length = min_section_length
        self.max_section_length = max_section_length

    def parse(self) -> tuple[list[str], list[str], list[str]]:
        ...

    def find_sections(self) -> bs4.element.ResultSet:
        ...

    def build_url(self, suffix: str) -> str:
        ...


class SphinxParser(Parser):
    def parse(self) -> tuple[list[str], list[str], list[str]]:
        found = self.find_sections()

        sections = []
        urls = []
        names = []
        for i in range(len(found)):
            section_found = found[i]

            section_soup = section_found.parent.parent
            section_href = section_soup.find_all("a", href=True, class_="headerlink")

            # If sections has subsections, keep only the part before the first subsection
            if len(section_href) > 1 and section_soup.section is not None:
                section_siblings = list(section_soup.section.previous_siblings)[::-1]
                section = parse_section(section_siblings)
            else:
                section = parse_section(section_soup.children)

            # Remove special characters, plus newlines in some url and section names.
            section = section.strip()
            url = section_found["href"].strip().replace("\n", "")
            name = section_found.parent.text.strip()[:-1].replace("\n", "")

            url = self.build_url(url)

            # If text is too long, split into chunks of equal sizes
            if len(section) > self.max_section_length:
                n_chunks = math.ceil(len(section) / float(self.max_section_length))
                separator_index = math.floor(len(section) / n_chunks)

                section_chunks = [section[separator_index * i : separator_index * (i + 1)] for i in range(n_chunks)]
                url_chunks = [url] * n_chunks
                name_chunks = [name] * n_chunks

                sections.extend(section_chunks)
                urls.extend(url_chunks)
                names.extend(name_chunks)
            # If text is not too short, add in 1 chunk
            elif len(section) > self.min_section_length:
                sections.append(section)
                urls.append(url)
                names.append(name)

        return sections, urls, names

    def find_sections(self) -> bs4.element.ResultSet:
        return self.soup.find_all("a", href=True, class_="headerlink")

    def build_url(self, suffix: str) -> str:
        return self.base_url + self.filename + suffix


class HuggingfaceParser(Parser):
    def parse(self) -> tuple[list[str], list[str], list[str]]:
        found = self.find_sections()

        sections = []
        urls = []
        names = []
        for i in range(len(found)):
            section_href = found[i].find("a", href=True, class_="header-link")

            section_nodes = []
            for element in found[i].find_next_siblings():
                if i + 1 < len(found) and element == found[i + 1]:
                    break
                section_nodes.append(element)
            section = parse_section(section_nodes)

            # Remove special characters, plus newlines in some url and section names.
            section = section.strip()
            url = section_href["href"].strip().replace("\n", "")
            name = found[i].text.strip().replace("\n", "")

            url = self.build_url(url)

            # If text is too long, split into chunks of equal sizes
            if len(section) > self.max_section_length:
                n_chunks = math.ceil(len(section) / float(self.max_section_length))
                separator_index = math.floor(len(section) / n_chunks)

                section_chunks = [section[separator_index * i : separator_index * (i + 1)] for i in range(n_chunks)]
                url_chunks = [url] * n_chunks
                name_chunks = [name] * n_chunks

                sections.extend(section_chunks)
                urls.extend(url_chunks)
                names.extend(name_chunks)
            # If text is not too short, add in 1 chunk
            elif len(section) > self.min_section_length:
                sections.append(section)
                urls.append(url)
                names.append(name)

        return sections, urls, names

    def find_sections(self) -> bs4.element.ResultSet:
        return self.soup.find_all(["h1", "h2", "h3"], class_="relative group")

    def build_url(self, suffix: str) -> str:
        return self.base_url + os.path.splitext(self.filename)[0] + suffix
