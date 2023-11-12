from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class PriceDataset(Dataset):

    def __init__(self, data, seq_len):
        assert len(data) >= seq_len, \
        f"Can not split data into sliding window of length {seq_len} with data of length {len(data)}"
        self.data = data
        self.seq_len = seq_len
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx+self.seq_len]).reshape(-1, self.seq_len).to(torch.float32)

    def __len__(self):
        return max(len(self.data)-self.seq_len, 0)

def train_val_split(data_indices, val_ratio=0.2):
  train_ratio = 1 - val_ratio
  last_train_index = int(np.round(len(data_indices) * train_ratio))
  return data_indices[:last_train_index], data_indices[last_train_index:]