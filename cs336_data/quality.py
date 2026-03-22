import fasttext
import os
from typing import Any

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
QUALITY_MODEL_PATH_FTZ = os.path.join(MODELS_DIR, 'quality_classifier.ftz')
QUALITY_MODEL_PATH_BIN = os.path.join(MODELS_DIR, 'quality_classifier.bin')

quality_model = None

def get_model_path():
    if os.path.exists(QUALITY_MODEL_PATH_FTZ):
        return QUALITY_MODEL_PATH_FTZ
    if os.path.exists(QUALITY_MODEL_PATH_BIN):
        return QUALITY_MODEL_PATH_BIN
    # fallback, will likely fail if missing
    return QUALITY_MODEL_PATH_FTZ

def classify_quality(text: str) -> tuple[Any, float]:
    global quality_model
    if quality_model is None:
        quality_model = fasttext.load_model(get_model_path())
        
    text = text.replace('\n', ' ')
    predictions, probabilities = quality_model.predict(text, k=1)
    
    label = predictions[0].replace('__label__', '')
    score = float(probabilities[0])
    
    return label, score
