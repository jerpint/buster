import pandas as pd

from buster.documents_manager import DeepLakeDocumentsManager

REQUIRED_COLUMNS = ["url", "title", "content", "source"]

# Read the csv
df = pd.read_csv("stackoverflow.csv")

# initialize our vector store from scratch
dm = DeepLakeDocumentsManager(vector_store_path="deeplake_store", overwrite=True, required_columns=REQUIRED_COLUMNS)

# Generate the embeddings for our documents and store them to the deeplake store
dm.add(df, csv_filename="embeddings.csv")
