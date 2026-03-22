import fasttext
import os
import urllib.request
from typing import Any

SERVER_MODEL_PATH = '/data/classifiers/lid.176.bin'
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
LOCAL_MODEL_PATH = os.path.join(MODELS_DIR, 'lid.176.bin')
LANGID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

model = None

def get_model_path(server_path, local_path, url):
    if os.path.exists(server_path):
        return server_path
    if os.path.exists(local_path):
        return local_path
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, local_path)
        return local_path
    except Exception as e:
        print(f"Warning: Failed to download model from {url}. Error: {e}")
        return local_path

def identify_language(text: str) -> tuple[Any, float]:
    global model
    if model is None:
        path = get_model_path(SERVER_MODEL_PATH, LOCAL_MODEL_PATH, LANGID_URL)
        model = fasttext.load_model(path)
    
    text = text.replace('\n', ' ')
    
    predictions, probabilities = model.predict(text, k=1)
    predicted_label = predictions[0]
    predicted_language = predicted_label.replace('__label__', '')
    score = float(probabilities[0])
    
    return predicted_language, score
