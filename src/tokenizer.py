import re
from typing import Dict


class SimpleTokenizer:

    def __init__(self, vocab: Dict):
        self.vocab = vocab
        self.ids_to_tokens = {v: k for k, v in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.vocab[token] for token in preprocessed if token in self.vocab]
        return ids

    def decode(self, ids):
        text = " ".join([self.ids_to_tokens[id] for id in ids])
        text = re.sub(r' ([,.?_!"()\']|--|\\s)', r'\\1', text)
        return text