"""
import json
import re

class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}

    def fit(self, text):
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        unique_words = sorted(set(words))
        for i, word in enumerate(unique_words):
            self.vocab[word] = i
            self.reverse_vocab[i] = word

    def encode(self, text):
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        return [self.vocab.get(word, len(self.vocab)) for word in words]

    def decode(self, tokens):
        return ' '.join([self.reverse_vocab.get(token, '<UNK>') for token in tokens])
    
    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.vocab, f, indent=2)
        print(f"Tokenizer saved to {filename}")

    def load(self, filename):
        try:
            with open(filename, 'r') as f:
                self.vocab = json.load(f)
            if not self.vocab:
                raise ValueError("Tokenizer file is empty")
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            print(f"Tokenizer loaded from {filename}")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error loading tokenizer: {e}")
            print("Initializing with a basic vocabulary...")
            self.initialize_basic_vocab()

    def initialize_basic_vocab(self):
        basic_words = ["<PAD>", "<UNK>", "the", "be", "to", "of", "and", "a", "at", "this", "but", "his", "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would", "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which", "go", "me"]
        self.vocab = {word: i for i, word in enumerate(basic_words)}
        self.reverse_vocab = {i: word for i, word in enumerate(basic_words)}
        """