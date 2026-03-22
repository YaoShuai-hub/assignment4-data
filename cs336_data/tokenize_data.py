import os
import argparse
import tiktoken
import numpy as np
import tqdm

def tokenize_directory(input_dir: str, output_file: str):
    """Tokenize all text files in input_dir using gpt2 tokenizer and save as numpy array."""
    # Using tiktoken's gpt2 tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    all_tokens = []
    
    # <|endoftext|> is usually 50256 in gpt2
    eot_token = enc.encode("<|endoftext|>", allowed_special="all")[0]
    
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.txt')]
    for file_path in tqdm.tqdm(files, desc="Tokenizing files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # We assume files contain multiple documents separated by double newlines or similar,
        # but the assignment asks to append <|endoftext|> at the end of each document.
        # Here we split by \n\n to emulate multiple docs, or just append to end of file if it's 1 doc.
        docs = [doc.strip() for doc in text.split('\n\n') if doc.strip()]
        for doc in docs:
            tokens = enc.encode(doc, allowed_special="all")
            tokens.append(eot_token)
            all_tokens.extend(tokens)
            
    print(f"Total tokens: {len(all_tokens)}")
    # Save as uint16 if max token id is < 65535, gpt2 vocab size is 50257
    print(f"Saving to {output_file}...")
    np_tokens = np.array(all_tokens, dtype=np.uint16)
    np_tokens.tofile(output_file)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    
    tokenize_directory(args.input_dir, args.output_file)
