import pandas as pd 
from ultis import *

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

# Label encoder
encode_dict = {}
def encode_cat(x):
  if x not in encode_dict.keys():
      encode_dict[x]=len(encode_dict)
  return encode_dict[x]
fake_raw['encoded_cat'] = fake_raw['category'].apply(lambda x: encode_cat(x))

# Drop rows that low frequency in category
value_counts = fake_raw['encoded_cat'].value_counts()
val_remove = value_counts[value_counts < 10].index.values
fake_raw['encoded_cat'].replace(val_remove, np.nan, inplace = True)
# fake_raw['encoded_cat'].loc[fake_raw['encoded_cat'].isin(val_remove)] = None
fake_raw.dropna(axis = 0, inplace = True, subset = ['encoded_cat'])
# Reset label dict
encode_dict = {}
fake_raw['encoded_cat'] = fake_raw['category'].apply(lambda x: encode_cat(x))
