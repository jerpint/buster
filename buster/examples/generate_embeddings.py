import click
import pandas as pd

from buster.documents_manager import DeepLakeDocumentsManager

REQUIRED_COLUMNS = ["url", "title", "content", "source"]


@click.command(
    help="This script processes a CSV file and generates embeddings. The CSV argument specifies the path to the input CSV file."
)
@click.argument("csv", metavar="<path_to_csv_file>")
def main(csv):
    # Read the csv
    df = pd.read_csv(csv)

    # initialize our vector store from scratch
    dm = DeepLakeDocumentsManager(vector_store_path="deeplake_store", overwrite=True, required_columns=REQUIRED_COLUMNS)

    # Generate the embeddings for our documents and store them to the deeplake store
    dm.add(df, csv_filename="embeddings.csv")

    # Save it to a zip file
    dm.to_zip()


if __name__ == "__main__":
    main()
