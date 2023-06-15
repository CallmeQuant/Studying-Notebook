from modules import *
from model import *
from ultis import *
from custom_dataset import *
from data_preprocessing import *

from transformers import *
from transformers.modeling_utils import *

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import torch
from torch import nn
import torch.utils.data
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold

# Reading pickle file to dataframe 
fake_raw_pkl = _load_pkl('/content/drive/My Drive/data/fake_raw_cleaned.pkl')
fake_raw = pd.read_pickle('fake_raw_cleaned.pkl')

# Define constants
hidden_size = 768
num_class = 16
EPOCHS = 5
BATCH_SIZE = 24 #6
N_FOLDS = 5
LR = 3e-5
CUR_DIR = os.path.dirname(os.getcwd())
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_PATH = '/content/drive/My Drive/Checkpoint/PhoBERT/Fake'
print(DEVICE)

# Configuring for BERT model
config = AutoConfig.from_pretrained('vinai/phobert-base-v2', output_hidden_states = True)
base_model = AutoModel.from_pretrained('vinai/phobert-base-v2', config = config)
model_bert = RobertaClassifier(base_model, hidden_size = hidden_size, num_class = num_class)
model_bert.to(DEVICE)

# Train-test-split and Stratifiying folds
df_train, df_test = np.split(fake_raw[['description','encoded_cat']].sample(frac=1, random_state=42),
                                     [int(.9*len(fake_raw[['description','encoded_cat']]))])
X_train = df_train['description'].values
y_train = df_train['encoded_cat'].values

splits = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=123).split(X_train, y_train))

# Training phase
result_fold = {}
for fold, (train_ids, val_ids) in enumerate(splits):
  print(f'Begin Training Fold {fold}')
  print('-'*80)
  batch_df_train = df_train.iloc[train_ids]
  batch_df_val = df_train.iloc[val_ids]

  train_loss, train_acc, val_loss, val_acc = train_on_epoch(model_bert, batch_df_train, 
                                                   batch_df_val, LR, EPOCHS)
  
  result_fold[fold] = 100 * (sum(val_acc)/len(val_acc))
  if fold == N_FOLDS - 1:
    print(f'Training {N_FOLDS} Folds Finished. Saving Trained Model!')

# Print fold results on training data
print(f'CROSS VALIDATION RESULTS FOR 5 FOLDS TRAIN SET')
print('-----------------------------------------------')
sum_acc = 0.0
for key, value in result_fold.items():
  print(f'Fold {key}: {value} %')
  sum_acc += value
print(f'Average accuracy validation: {sum_acc/len(result_fold.items())} %')

# Predicting on test set
predict_ensem = []
avg_acc = []
for fold in range(N_FOLDS):
  print(f"Predicting for fold {fold}")
  label_pred, acc_test = evaluate(model_bert, df_test)
  print('-'*80)
  predict_ensem.append(label_pred)
  avg_acc.append(acc_test)

print(f'CROSS VALIDATION RESULTS FOR 5 FOLDS TEST SET')
print('----------------------------------------------')

print(f'Average accuracy test: {sum(avg_acc)/len(avg_acc)} %')
