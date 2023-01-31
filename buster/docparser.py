import glob
import math
import os

import bs4
import pandas as pd
import tiktoken
from bs4 import BeautifulSoup
from openai.embeddings_utils import get_embedding

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_ENCODING = "cl100k_base"  # this the encoding for text-embedding-ada-002


BASE_URL_MILA = "https://docs.mila.quebec/"
BASE_URL_ORION = "https://orion.readthedocs.io/en/stable/"


def parse_section(nodes: list[bs4.element.NavigableString]) -> str:
    section = []
    for node in nodes:
        if node.name == "table":
            node_text = pd.read_html(node.prettify())[0].to_markdown(index=False, tablefmt="github")
        else:
            node_text = node.text
        section.append(node_text)
    section = "".join(section)[1:]

    return section


def get_all_documents(root_dir: str, base_url: str, max_section_length: int = 2000) -> pd.DataFrame:
    """Parse all HTML files in `root_dir`, and extract all sections.

    Sections are broken into subsections if they are longer than `max_section_length`.
    Sections correspond to `section` HTML tags that have a headerlink attached.
    """
    files = glob.glob("**/*.html", root_dir=root_dir, recursive=True)

    def get_all_subsections(soup: BeautifulSoup) -> tuple[list[str], list[str], list[str]]:
        found = soup.find_all("a", href=True, class_="headerlink")

        sections = []
        urls = []
        names = []
        for section_found in found:
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

            # If text is too long, split into chunks of equal sizes
            if len(section) > max_section_length:
                n_chunks = math.ceil(len(section) / float(max_section_length))
                separator_index = math.floor(len(section) / n_chunks)

                section_chunks = [section[separator_index * i : separator_index * (i + 1)] for i in range(n_chunks)]
                url_chunks = [url] * n_chunks
                name_chunks = [name] * n_chunks

                sections.extend(section_chunks)
                urls.extend(url_chunks)
                names.extend(name_chunks)
            else:
                sections.append(section)
                urls.append(url)
                names.append(name)

        return sections, urls, names

    sections = []
    urls = []
    names = []
    for file in files:
        filepath = os.path.join(root_dir, file)
        with open(filepath, "r") as f:
            source = f.read()

        soup = BeautifulSoup(source, "html.parser")
        sections_file, urls_file, names_file = get_all_subsections(soup)
        sections.extend(sections_file)

        urls_file = [base_url + file + url for url in urls_file]
        urls.extend(urls_file)

        names.extend(names_file)

    documents_df = pd.DataFrame.from_dict({"name": names, "url": urls, "text": sections})

    return documents_df


def write_documents(filepath: str, documents_df: pd.DataFrame):
    documents_df.to_csv(filepath, index=False)


def read_documents(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def compute_n_tokens(df: pd.DataFrame) -> pd.DataFrame:
    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    df["n_tokens"] = df.text.apply(lambda x: len(encoding.encode(x)))
    return df


def precompute_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    df["embedding"] = df.text.apply(lambda x: get_embedding(x, engine=EMBEDDING_MODEL))
    return df


def generate_embeddings(filepath: str, output_csv: str) -> pd.DataFrame:
    # Get all documents and precompute their embeddings
    df = read_documents(filepath)
    df = compute_n_tokens(df)
    df = precompute_embeddings(df)
    write_documents(output_csv, df)
    return df


if __name__ == "__main__":
    root_dir = "/home/hadrien/perso/mila-docs/output/"
    save_filepath = "data/documents.csv"

    # How to write
    documents_df = get_all_documents(root_dir)
    write_documents(save_filepath, documents_df)

    # How to load
    documents_df = read_documents(save_filepath)

    # precompute the document embeddings
    df = generate_embeddings(filepath=save_filepath, output_csv="data/document_embeddings.csv")
