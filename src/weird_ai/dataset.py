import torch
from torch.utils.data import Dataset

class LyricsDataset:
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, index):
        # TODO:
        # Get the input/output token sequences
        start = index
        end = index + self.block_size

        # Calculate x by grabbing the sublist of tokens starting at the index up to the block_size
        x_tokens = self.tokens[start:end]
        # Calculate y by grabbing the sublist of tokens starting at index + 1 up to block_size + 1
        y_tokens = self.tokens[start + 1 : end + 1]

        x = torch.tensor(x_tokens)
        y = torch.tensor(y_tokens)
        
        return x, y