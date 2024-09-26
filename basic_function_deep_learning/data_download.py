
import os
import zipfile

from pathlib import Path

import requests

def download_data(source: str,
                  destination: str,
                  remove_source: bool = True) -> Path:
    """
    Downloads a zipped dataset from source and unzips to destination.
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, create it
    if image_path.is_dir():
      print(f"[INFO] {image_path} directory already exists, skipping download.")
    else:
      print(f"[INFO] Did not find {image_path} directory, creating one...")
      image_path.mkdir(parents=True, exist_ok=True)

      # Download the target
      target_file = Path(source).name
      with open(data_path / target_file, "wb") as f:
        request = requests.get(source)
        print(f"[INFO] Downloading {target_file}")
        f.write(request.content)

      # Unzip target file
      with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
        print(f"[INFO] Unzipping {target_file} data...")
        zip_ref.extractall(image_path)

      # Remove .zip file if needed
      if remove_source:
        os.remove(data_path / target_file)

    return image_path
