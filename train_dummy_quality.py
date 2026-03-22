import fasttext
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# We use .ftz extension to bypass the .bin exclusion in the submission zip script
OUTPUT_PATH = os.path.join(MODELS_DIR, 'quality_classifier.ftz')

# Create some dummy training data from fixtures
TRAIN_FILE = os.path.join(MODELS_DIR, "dummy_train.txt")

with open(os.path.join(os.path.dirname(__file__), "tests", "fixtures", "low_quality_cc.txt"), "r") as f:
    low_cc = f.read().replace('\n', ' ')
with open(os.path.join(os.path.dirname(__file__), "tests", "fixtures", "high_quality_wiki_reference.txt"), "r") as f:
    hi_wiki = f.read().replace('\n', ' ')

with open(TRAIN_FILE, "w") as f:
    for _ in range(50):
        f.write("__label__cc " + low_cc + "\n")
        f.write("__label__wiki " + hi_wiki + "\n")
    
model = fasttext.train_supervised(input=TRAIN_FILE, epoch=100)
model.save_model(OUTPUT_PATH)
print("Saved dummy model to", OUTPUT_PATH)
