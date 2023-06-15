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

# Reading data to dataframe type from xlsx
df_raw = pd.read_excel('drive/My Drive/data/Transaction_Categorization_Data_2.xlsx', sheet_name = None)
fake_raw = preprocess_xlsx(df_raw['Fake Data'])
real_raw = preprocess_xlsx(df_raw['Real Data'], is_fake = False)

# Preprocessing fake data
fake_raw['loại_thu_nhập'] = fake_raw['loại_thu_nhập'].fillna(fake_raw['loại_chi_trả'])
fake_raw = fake_raw.drop(['loại_chi_trả'], axis = 1)
fake_raw = fake_raw.rename(columns = {'loại_thu_nhập':'category'})
fake_raw['category'] = fake_raw['category'].str.lower()
fake_raw = fake_raw[~fake_raw['category'].str.contains('tiền nhà, bảo hiểm|tiền điện, tiền nước|wi-fi, khác')]
fake_raw['category'] = fake_raw['category'].replace(' ', '_', regex = True)
