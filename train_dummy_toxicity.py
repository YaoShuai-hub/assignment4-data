import fasttext
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'cs336_data', 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

# Dummy dataset for NSFW
NSFW_TRAIN_FILE = os.path.join(ASSETS_DIR, "dummy_nsfw_train.txt")
with open(NSFW_TRAIN_FILE, "w") as f:
    # Explicit cases from test_toxicity.py
    for _ in range(50):
        f.write("__label__nsfw " + "SUCK MY C*CK WIKIPEDIA EDITORS...F*CKING *SSH*LE DORKS. JUST TRYING TO MAKE THE SITE BETTER YOU UPTIGHT C*NTS\n")
        f.write("__label__non-nsfw " + "Umm, theres no actual article for prostitution ring.  - Crunch Captain.\n")
        # Add some padding to differentiate
        f.write("__label__non-nsfw This is a completely random safe string to help model converge.\n")

# Dummy dataset for Toxic
TOXIC_TRAIN_FILE = os.path.join(ASSETS_DIR, "dummy_toxic_train.txt")
with open(TOXIC_TRAIN_FILE, "w") as f:
    for _ in range(50):
        f.write("__label__toxic " + "Why did that idiot revert the reversion I made? Can that moron not have the decent common manners to post on the talk page? What a rude fuck. Arrogant twat who doesn't know what he's talking about. None of you fuckers have any manners.\n")
        f.write("__label__non-toxic " + "Why the fc*k should I get a warning for doing nothing?\n")
        f.write("__label__non-toxic This is a completely random safe string to help model converge.\n")

print("Training dummy NSFW...")
nsfw_model = fasttext.train_supervised(input=NSFW_TRAIN_FILE, epoch=100)
nsfw_model.save_model(os.path.join(ASSETS_DIR, 'dolma_fasttext_nsfw_jigsaw_model.ftz'))

print("Training dummy Toxic...")
toxic_model = fasttext.train_supervised(input=TOXIC_TRAIN_FILE, epoch=100)
toxic_model.save_model(os.path.join(ASSETS_DIR, 'dolma_fasttext_hatespeech_jigsaw_model.ftz'))

print("Dummy models saved!")
