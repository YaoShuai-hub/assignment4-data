import urllib.request
import os

os.makedirs('models', exist_ok=True)
url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
output_path = "models/lid.176.bin"

if not os.path.exists(output_path):
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, output_path)
    print("Download complete.")
else:
    print("Model already exists.")
