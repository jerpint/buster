import pandas as pd

from buster.documents_manager import DeepLakeDocumentsManager

# Read the csv
df = pd.read_csv("stackoverflow.csv")

# Generate the embeddings for our documents and store them in a deeplake format
dm = DeepLakeDocumentsManager(vector_store_path="deeplake_store", overwrite=True)
dm.add(df)
