import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from datetime import datetime, timedelta
from crawler import get_stocks_price

class Dataset:
  """Dataset class for modelling"""
  def __init__(self, end_date, stock_list, train_ratio, train_len, pred_len, batch_size):
    # batch_size = num_obs + 1
    self.data = self.get_data(end_date, stock_list = stock_list)
    self.data_values = torch.tensor(self.data.values, dtype = torch.float32)
    self.data_index = self.data.index.to_series().astype(int).values  # Convert Timestamp to int
    self.train_ratio = train_ratio
    self.train_len = train_len
    self.pred_len = pred_len
    self.seq_len = train_len + pred_len
    self.batch_size = batch_size
    self.stock_names = self.data.columns.tolist()
    self.num_stock = len(self.data.columns.tolist())

  def create_dataset(self, data_values, data_index, seq_len):
    # shape: batch x seq_len x num_features/num_stocks
    dataset = data_values.unfold(0, seq_len, 1).permute(0, 2, 1)
    times = [data_index[i : seq_len + i] for i in range(len(data_values) - seq_len + 1)]

    # Separate the input and target data
    x = dataset[:, :-self.pred_len, :]
    y = dataset[:, -self.pred_len:, :]

    # Separate the time indices for x and y
    times_x = [t[:-self.pred_len] for t in times]
    times_y = [t[-self.pred_len:] for t in times]

    return x, y, times_x, times_y, times

  def create_loader(self, split_idx = None):
    if split_idx is None:
      split_idx = int(len(self.data_values) * self.train_ratio)
    data_train_values, data_test_values = torch.split(self.data_values, split_idx, dim=0)
    data_train_index, data_test_index = torch.split(torch.tensor(self.data_index), split_idx, dim=0)

    x_train, y_train, times_train, _, _ = self.create_dataset(data_train_values,
                                            data_train_index, self.seq_len)
    x_test, y_test, times_test, _, _ = self.create_dataset(data_test_values,
                                          data_test_index, self.seq_len)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset,
                              batch_size=self.batch_size,
                              shuffle=True,
                              drop_last=True,)
    test_loader = DataLoader(test_dataset,
                              batch_size=self.batch_size,
                              shuffle=False,
                             drop_last=True,)

    return train_loader, test_loader, times_train, times_test

  def get_data(self, end_date,
               start_date = None,
               stock_list = ['CII','DIG','HPG',"HT1",'HSG',"GAS","GVR",'TPB','TCB','MSN']):
    if start_date is None:
      start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days = 7 * 365)).strftime('%Y-%m-%d')
    df = get_stocks_price(stock_list,
                          dataframe_type = 'long',
                          start_date = start_date, end_date = end_date)

    # df.columns = df.columns.droplevel(1)
    df = df.loc[:, ['ticker', 'close']].reset_index()
    df['dow'] = df.date.apply(lambda x: x.dayofweek)
    ## just select working days
    df = df[(df.dow<=4)&(df.dow>=0)]
    df = df.drop(['dow'],axis=1)
    df = df.pivot_table(index='date', columns='ticker')
    ## select tickers not nan in final day
    columns = df.close.columns[~df.close.iloc[-1].isna()]
    df = df.iloc[:, df.columns.get_level_values(1).isin(columns)]
    df.close = df.close.interpolate(method='linear',limit_area='inside',limit_direction='both', axis=0)
    df.dropna(inplace = True)

    df_log_ret = np.log(df.close/df.close.shift(1)).fillna(1e-2)
    df_log_ret = df_log_ret.interpolate(method='linear',limit_area="inside",limit_direction='both', axis=0)

    return df_log_ret

  def get(self, train_mode = True):
    train_loader, test_loader, _, _ = self.create_loader()
    if train_mode:
      return train_loader
    else:
      return test_loader
