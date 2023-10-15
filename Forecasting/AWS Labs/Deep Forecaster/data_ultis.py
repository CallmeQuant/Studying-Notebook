
import numpy as np
import pandas as pd
from datetime import date
import torch
import os
import random
from typing import *
from progressbar import *

def load_and_process_data(data_path: str):
    data = (
        pd.read_csv(data_path, sep=",", header=0, parse_dates=['date'])
        .astype({'date': 'datetime64[ns]'})
        .assign(
            year=(lambda x: x['date'].dt.year),
            day_of_week=(lambda x: x['date'].dt.dayofweek))
    )

    data = data.loc[(data["date"] >= pd.to_datetime(date(2014, 1, 1))) &
                    (data["date"] <= pd.to_datetime(date(2014, 3, 1)))]
    data.index = pd.RangeIndex(len(data.index))  # reset index due to compare date

    # Create feature matrix
    X = np.c_[np.asarray(data['hour']), np.asarray(data['day_of_week'])]
    num_feats = X.shape[1]  # features = ['hour', 'day_of_week']
    num_periods = len(data)

    X = np.asarray(X).reshape(
        (-1, num_periods, num_feats))  # To ensure the shape of univariate series is (1, num_periods, num_feats)
    y = np.asarray(data["MT_200"]).reshape((-1, num_periods))
    return data, X, y, num_feats, num_periods

def split_train_test(X: np.array, y: np.array, split_ratio: float = 0.7):
    num_ts, num_periods, num_feats = X.shape
    train_length = int(num_periods * split_ratio)
    X_train = X[:, :train_length, :]
    y_train = y[:, :train_length]
    X_test = X[:, train_length:, :]
    y_test = y[:, train_length:]
    return X_train, y_train, X_test, y_test
def create_batch(X, y, prior, seq_len, batch_size):
    '''
    Function to create batch for training.
    For univariate case, the sampled batch will be the same repeatedly.
    For multivariate time series, the sampled batch will contain numbers of different time series
    that is equal to the batch size (for example, 64). This ensures the diversity of training; thereby
    leads to more generalizable.

    Parameters
    -----------

    X (array like): shape (num_samples, num_periods, num_features)
    y (array like): shape (num_samples, num_periods)
    piror (int)
    seq_len (int): sequence/encoder/decoder length
    '''
    num_ts, num_periods, _ = X.shape

    # If batch size is less than number of samples, using full batch. Otherwise, use batch size
    if batch_size < num_ts:
        print(f'Using full batch of size {num_ts} for training')
    else:
        print(f'Using mini-batch of size {batch_size} for training')
    batch_size = min(num_ts, batch_size)

    # Pre-calculate the range outside the choice function for efficiency
    time_range = range(prior, num_periods - seq_len)
    time_split = random.choice(time_range)

    batch_range = range(num_ts)
    batch = random.sample(batch_range, batch_size)

    # Can use empty array then store sliced values for better pre-allocation memory
    # Use slicing to get the batches
    X_train_batch = X[batch, time_split - prior :time_split, :]
    y_train_batch = y[batch, time_split - prior :time_split]

    Xf = X[batch, time_split:time_split + seq_len, :]
    yf = y[batch, time_split:time_split + seq_len]

    return X_train_batch, y_train_batch, Xf, yf


