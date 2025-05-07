from datasets import load_from_disk
from gensim.models import Word2Vec
import os

langs = ["ZTurkish"]
strategies = ["SUBWORDCORRECTEDBPE5k"]

output_path = rf"C:\Users\jinfa\Desktop\Research Dr. Mani\ZTurkish Word2Vec"

def make_model(lang, strategy):
    dataset = load_from_disk(rf"C:\Users\jinfa\OneDrive\Desktop\Research Dr. Mani\{lang} Tokenized\{strategy}")
    tokens = dataset['train']['tokens']

    print(f"Training Word2Vec model for {strategy}...")
    model = Word2Vec(sentences=tokens, vector_size=150, window=5, min_count=1, workers=4)
    
    # Save the model
    model_path = os.path.join(output_path, f"{lang}_{strategy}_word2vec.model")
    model.save(model_path)
    print(f"Model saved to {model_path}")

for lang in langs:
    for strategy in strategies:
        make_model(lang, strategy)



