import json
import re
from typing import List, Union

class Tokenizer:
    def __init__(self, case_sensitive: bool = False, max_vocab_size: int = None):
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.reverse_vocab = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.next_id = 4
        self.case_sensitive = case_sensitive
        self.max_vocab_size = max_vocab_size

    def fit(self, text: str):
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        if not self.case_sensitive:
            words = [word.lower() for word in words]

        unique_words = sorted(set(words))
        if self.max_vocab_size:
            unique_words = unique_words[:self.max_vocab_size]
        
        for word in unique_words:
            if word not in self.vocab:
                self.vocab[word] = self.next_id
                self.reverse_vocab[self.next_id] = word
                self.next_id += 1

    def encode(self, text: str) -> List[int]:
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        if not self.case_sensitive:
            words = [word.lower() for word in words]
        return [self.vocab.get(word, self.vocab['<UNK>']) for word in words]

    def decode(self, tokens: Union[List[int], int]) -> str:
        if isinstance(tokens, int):
            tokens = [tokens]
        return ' '.join([self.reverse_vocab.get(token, '<UNK>') for token in tokens])

    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, filename: str):
        try:
            with open(filename, 'w') as f:
                json.dump(self.vocab, f, indent=2)
            print(f"Tokenizer saved to {filename}")
        except IOError as e:
            print(f"Error saving tokenizer: {e}")

    def load(self, filename: str):
        try:
            with open(filename, 'r') as f:
                self.vocab = json.load(f)
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            print(f"Tokenizer loaded from {filename}")
        except IOError as e:
            print(f"Error loading tokenizer: {e}")

# Example usage
tokenizer = Tokenizer(case_sensitive=False, max_vocab_size=1000)
tokenizer.fit("Hello world! This is a simple tokenizer.")
tokens = tokenizer.encode("Hello world!")
decoded_text = tokenizer.decode(tokens)
print(tokens, decoded_text)
