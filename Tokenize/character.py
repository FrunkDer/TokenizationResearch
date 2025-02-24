from datasets import load_dataset

lang = "ZTurkish"
# Load your .txt files
dataset = load_dataset(
    "text", 
    data_files={
        "train": rf"C:\Users\jinfa\OneDrive\Desktop\Research Dr. Mani\{lang}10k\article_*.txt"
    }
)

def char_tokenize(examples):
    """
    Converts each text in examples["text"] into a list of individual 
    characters, removing all whitespace characters.
    """
    tokenized_texts = []
    for text in examples["text"]:
        # Remove all whitespace (spaces, tabs, newlines) by splitting and rejoining
        text_no_whitespace = "".join(text.split())
        # Convert the remaining string into a list of individual chars
        tokens = list(text_no_whitespace)
        tokenized_texts.append(tokens)
    
    return {"tokens": tokenized_texts}

tokenized_dataset = dataset.map(char_tokenize, batched=True)

# Choose a save path that includes the "Finnish Tokenized" subfolder
save_path = rf"C:\Users\jinfa\OneDrive\Desktop\Research Dr. Mani\{lang} Tokenized\Char"

tokenized_dataset.save_to_disk(save_path)
