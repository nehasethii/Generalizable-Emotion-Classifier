import os
import zipfile
import requests
from tqdm import tqdm

# Directory to store embeddings
embedding_dir = "embeddings"
os.makedirs(embedding_dir, exist_ok=True)

# Download link for only glove.6B.zip (we’ll extract just the 100d file)
url = "http://nlp.stanford.edu/data/glove.6B.zip"
zip_path = os.path.join(embedding_dir, "glove.6B.zip")

# Download with progress bar
if not os.path.exists(zip_path):
    print("Downloading GloVe embeddings (~822MB, only 100d will be extracted)...")
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(zip_path, 'wb') as file, tqdm(
        desc="Downloading", total=total, unit='B', unit_scale=True, unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
else:
    print("GloVe zip already exists, skipping download.")

# Extract only 100d file
print("Extracting only glove.6B.100d.txt ...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for file in zip_ref.namelist():
        if file == "glove.6B.100d.txt":
            zip_ref.extract(file, embedding_dir)

print("✅ Done! File saved at:", os.path.join(embedding_dir, "glove.6B.100d.txt"))
