import re
import pandas as pd 
import pickle

def preprocess_xlsx(df, is_fake = True):
  df.columns = df.columns.str.lower()
  df.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
  df.drop(['stt'], axis = 1, inplace = True)
  df.reset_index(inplace = True, drop = True)
  if is_fake:
    df.rename(columns = {'chú_thích_giao_dịch':'description'}, inplace = True)
    df.dropna(subset = ['description'], inplace = True)
  else:
    df.dropna(subset = ['description'], inplace = True)
  return df

def encode_cat(x):
  if x not in encode_dict.keys():
      encode_dict[x]=len(encode_dict)
  return encode_dict[x]

def decode_cat(x):
  for key, value in encode_dict.items():
    if x == value:
      return key
    
def normalize_data(row):
  # Lowering sentence
  row = row.lower()
  # Remove . ? , at index final
  row = re.sub(r"[\.,\?]+$-", "", str(row))
  # Remove all . , " ... in sentences
  row = row.replace(",", " ").replace(".", " ") \
      .replace(";", " ").replace("“", " ") \
      .replace(":", " ").replace("”", " ") \
      .replace('"', " ").replace("'", " ") \
      .replace("!", " ").replace("?", " ") \
      .replace("-", " ").replace("?", " ")
  # Remove date in sentences
  row = re.sub(r"^(?:(?:[0-9]{2}[\/,]){2}[0-9]{2,4})$|\d+/\d+", "", str(row))
  return row

def _save_pkl(path, obj):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)

def _load_pkl(path):
  with open(path, 'rb') as f:
    obj = pickle.load(f)
  return obj
