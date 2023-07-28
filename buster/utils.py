import os
import urllib.request
import zipfile


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


def zip_contents(input_path, output_path):
    """
    Zips the entire contents of a given path to a custom output path.

    Authored by ChatGPT

    Args:
        input_path (str): The path of the directory to be zipped.
        output_path (str): The path where the zip file will be created.

    Returns:
        str: The path of the created zip file.
    """
    if not os.path.exists(input_path):
        raise ValueError("The specified input path does not exist.")

    zip_file_name = f"{os.path.basename(input_path)}.zip"
    zip_file_path = os.path.join(output_path, zip_file_name)

    with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(input_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, input_path)
                zipf.write(file_path, arcname=arcname)

    return zip_file_path


def extract_zip(zip_file_path, output_path):
    """
    Extracts the contents of a zip file to a custom output path.

    Authored by ChatGPT

    Args:
        zip_file_path (str): The path of the zip file to be extracted.
        output_path (str): The path where the zip contents will be extracted.

    Returns:
        str: The path of the directory where the zip contents are extracted.
    """
    if not os.path.exists(zip_file_path):
        raise ValueError("The specified zip file does not exist.")

    with zipfile.ZipFile(zip_file_path, "r") as zipf:
        zipf.extractall(output_path)

    return output_path
