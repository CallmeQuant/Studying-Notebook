import torch.utils.data
from transformers import *

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

class Dataset(torch.utils.data.Dataset):
  
  def __init__(self, df):
    self.labels = [label for label in df['encoded_cat']]
    self.texts = [tokenizer(text,
                           padding='max_length', max_length = 258, truncation=True,
                            return_tensors="pt") for text in df['description']]

  def classes(self):
    return self.labels

  def __len__(self):
     return len(self.labels)

  def get_batch_labels(self, idx):
     # Fetch a batch of labels
     return np.array(self.labels[idx])

  def get_batch_texts(self, idx):
     # Fetch a batch of inputs
     return self.texts[idx]

  def __getitem__(self, idx):

     batch_texts = self.get_batch_texts(idx)
     batch_y = self.get_batch_labels(idx)

     return batch_texts, batch_y
