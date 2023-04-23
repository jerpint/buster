import os
import fitz
import logging
import click
from dataclasses import dataclass
import pandas as pd
from docparser import generate_embeddings
from buster.utils import get_documents_manager_from_extension

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_ENCODING = "cl100k_base"  # this the encoding for text-embedding-ada-002


@dataclass
class Chunk:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    page_num: int

    def merge(self, next_chunk: "Chunk"):
        """ Merge two chunk together """
        return Chunk(x0=min(self.x0, next_chunk.x0),
                     y0=min(self.y0, next_chunk.y0),
                     x1=max(self.x1, next_chunk.x1),
                     y1=max(self.y1, next_chunk.y1),
                     text=self.text + ' ' + next_chunk.text,
                     page_num=self.page_num)
        

class PDFParser:
    """ A parser that return each section as one page. """
    def __init__(self, pdf_path, chunk_size):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
    
    def parse(self):
        assert os.path.exists(self.pdf_path), f"{self.pdf_path} is not a local file"
        doc = fitz.open(self.pdf_path)
        pdflines = []
        for idx, page in enumerate(doc):
            pagedict = page.get_text('dict', sort=True)
            for blk in pagedict['blocks']:
                # Skip image
                if blk['type'] != 0:
                    continue
                
                for line in blk['lines']:
                    
                    # Construct line from spans
                    linetext = []
                    for span in line['spans']:
                        if 'text' in span:
                            span_text = span['text']
                        elif 'chars' in span:
                            span_text = ''.join([char["c"] for char in span['chars']])
                        else:
                            raise ValueError('Impossible span')
                        
                        linetext.append(span_text)
                    linetext = ' '.join(linetext)
                    pdfline = Chunk(
                        x0=line['bbox'][0],
                        y0=line['bbox'][1],
                        x1=line['bbox'][2],
                        y1=line['bbox'][3],
                        text=linetext,
                        page_num=idx+1)
                    pdflines.append(pdfline)
                    
        # Merge PDFlines into actual chunk
        chunks = []
        curr_chunk = None
        for pdfline in pdflines:
            if curr_chunk is None:
                curr_chunk = pdfline
                
            # Insert into chunks
            if len(curr_chunk.text.split()) >= self.chunk_size:
                chunks.append(curr_chunk)
                curr_chunk = None
            
            # Merge
            else:
                curr_chunk = curr_chunk.merge(pdfline)
        
        if curr_chunk is not None:
            chunks.append(curr_chunk)
        return chunks


@click.command()
@click.argument("pdf_path")
@click.option(
    "--output-filepath", default="documents.db", help='Where your database will be saved. Default is "documents.db"'
)
@click.option(
    "--chunk_size", default=500, help="Number of maximum allowed words per document, excess is trimmed. Default is 500"
)
@click.option(
    "--embeddings-engine", default=EMBEDDING_MODEL, help=f"Embedding model to use. Default is {EMBEDDING_MODEL}"
)
def main(pdf_path: str, output_filepath: str, chunk_size: int, embeddings_engine: str):
    parser = PDFParser(pdf_path=pdf_path, chunk_size=chunk_size)
    chunks = parser.parse()
    # Building CSV
    rows = [ ['pdf', f"page {blk.page_num}", blk.text, 'file://' + os.path.realpath(pdf_path) + f"#page={blk.page_num}"] for blk in chunks]
    documents = pd.DataFrame(rows, columns=['source', 'title', 'content', 'url'])
    logger.info(f"Documents saved to: {output_filepath}")
    documents_manager = get_documents_manager_from_extension(output_filepath)(output_filepath)
    documents = generate_embeddings(documents, documents_manager, chunk_size, embeddings_engine)

if __name__ == "__main__":
    main()
