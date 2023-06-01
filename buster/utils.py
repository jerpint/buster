import os
import urllib.request


def get_file_extension(filepath: str) -> str:
    return os.path.splitext(filepath)[1]


def download_db(db_url: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, "documents.db")
    if not os.path.exists(fname):
        print(f"Downloading db file from {db_url} to {fname}...")
        urllib.request.urlretrieve(db_url, fname)
        print("Downloaded.")
    else:
        print("File already exists. Skipping.")
    return fname
