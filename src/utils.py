from torch.nn import functional as F
from torch.utils.data import Dataset
import numpy as np
import random
import torch
import re


stoi = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '\n': 10, '000000000000': 11}
itos = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '\n', 11: '000000000000'}

tok_chars = re.compile(r'000000000000|\d{1}|\n')

def encode(text, stoi, tokenizer):
    matches = tokenizer.findall(text)
    return [stoi[c] for c in matches if c in stoi]

def decode(encoded, itos):
    return ''.join([itos[i] for i in encoded])


class Dataset:
    def __init__(self, data, ctx_len, epoch_length_fixed, batch_size):
        print('tokenizing data...')
        self.ctx_len = ctx_len
        self.epoch_length_fixed = epoch_length_fixed
        self.batch_size = batch_size
        self.start_token = '000000000000'
        self.tokenizer = tok_chars
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(stoi)
        print('vocab size:', self.vocab_size)
        self.data = encode(data, self.stoi, self.tokenizer)
        self.data_size = len(self.data)
        print(f'data has {self.data_size} tokens')

    def __len__(self):
        return self.epoch_length_fixed

    def __getitem__(self, idx):
        cues = []
        idx_randm = random.randint(0, len(self.data) - (self.ctx_len) * (2 * self.batch_size))
        i = idx_randm

        while True:
            if self.data[i] == self.stoi[self.start_token]:
                cues = [i]
                break
            else:
                i = (i + 1) % len(self.data)

        if not cues:
            return None

        start_idx = cues[0]
        dix = self.data[start_idx : start_idx + self.ctx_len + 2]
        x = torch.tensor(dix[:-1][:self.ctx_len], dtype=torch.int64)
        y = torch.tensor(dix[1:][:self.ctx_len], dtype=torch.int64)

        return x, y


class TOKENIZER():
    def __init__(self):
        self.tokenizer = tok_chars
        self.stoi = stoi
        self.itos = itos
        self.vocab_size = len(self.stoi)

    def encode(self, text):
        matches = self.tokenizer.findall(text)
        return [self.stoi[c] for c in matches if c in self.stoi]

    def decode(self, encoded):
        return ''.join([self.itos[i] for i in encoded])

    def sample_logits(self, out, x, ctx_len, temperature=1.0, top_k=50):
        probs = F.softmax(torch.tensor(out), dim=-1)
        
        if top_k > 0:
            top_k = min(top_k, probs.size(-1))
            sorted_probs, sorted_indices = torch.topk(probs, top_k)
            probs.fill_(0)
            probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
        
        if temperature != 1.0:
            probs = probs.pow(1.0 / temperature)

        return torch.multinomial(probs, num_samples=1)[0]

