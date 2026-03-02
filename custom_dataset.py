import torch
from tiktoken import Encoding
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, text: str, tokenizer: Encoding, seq_len: int):
        self.samples = []
        conversations = text.split("<|eos|>")
        for conv in conversations:
            conv = conv.strip()
            if len(conv) > 0:
                tokens = tokenizer.encode(conv + "<|eos|>")
                self.samples.append(torch.tensor(tokens, dtype=torch.long))

        self.seq_len = seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]

        if len(tokens) > self.seq_len:
            tokens = tokens[: self.seq_len]

        x = tokens[:-1]
        y = tokens[1:]

        return x, y
