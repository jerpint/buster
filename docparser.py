import glob
import os
import pickle

import pandas as pd
import tiktoken
from bs4 import BeautifulSoup
from openai.embeddings_utils import cosine_similarity, get_embedding

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_ENCODING = "cl100k_base"  # this the encoding for text-embedding-ada-002


def get_all_sections(root_dir: str, max_section_length: int = 3000) -> list[str]:
    """Parse all HTML files in `root_dir`, and extract all sections.

    Sections are broken into subsections if they are longer than `max_section_length`.
    Sections correspond to h2 HTML tags, and move on to h3 then h4 if needed.
    """
    files = glob.glob("*.html", root_dir=root_dir)

    selector = "section > section"

    # Recurse until sections are small enough
    def get_all_subsections(soup, selector: str) -> list[str]:
        found = soup.select(selector)
        data = [x.text.split(";")[-1].strip() for x in found]

        sections = []
        for i, section in enumerate(data):
            if len(section) > max_section_length:
                sections.extend(get_all_subsections(found[i], selector + " > section"))
            else:
                sections.append(section)

        return sections

    sections = []
    for file in files:
        filepath = os.path.join(root_dir, file)
        with open(filepath, "r") as file:
            source = file.read()

        soup = BeautifulSoup(source, "html.parser")
        sections.extend(get_all_subsections(soup, selector))

    return sections


def write_sections(filepath: str, sections: list[str]):
    with open(filepath, "wb") as f:
        pickle.dump(sections, f)


def read_sections(filepath: str) -> list[str]:
    with open(filepath, "rb") as fp:
        sections = pickle.load(fp)

    return sections


def load_documents(fname: str) -> pd.DataFrame:
    df = pd.DataFrame()

    with open(fname, "rb") as fp:
        documents = pickle.load(fp)
    df["documents"] = documents
    return df


def compute_n_tokens(df: pd.DataFrame) -> pd.DataFrame:
    encoding = tiktoken.get_encoding(EMBEDDING_ENCODING)
    df["n_tokens"] = df.documents.apply(lambda x: len(encoding.encode(x)))
    return df


def precompute_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    df["embedding"] = df.documents.apply(lambda x: get_embedding(x, engine=EMBEDDING_MODEL))
    return df


def generate_embeddings(filepath: str, output_csv: str) -> pd.DataFrame:
    # Get all documents and precompute their embeddings
    df = load_documents(filepath)
    df = compute_n_tokens(df)
    df = precompute_embeddings(df)
    df.to_csv(output_csv)
    return df


if __name__ == "__main__":
    root_dir = "/home/hadrien/perso/mila-docs/output/"
    save_filepath = os.path.join(root_dir, "sections.pkl")

    # How to write
    sections = get_all_sections(root_dir)
    write_sections(save_filepath, sections)

    # How to load
    sections = read_sections(save_filepath)

    # precopmute the document embeddings
    df = generate_embeddings(filepath=save_filepath, output_csv="data/document_embeddings.csv")
