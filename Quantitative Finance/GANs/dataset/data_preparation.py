from datetime import datetime, timedelta
from transform.gaussianize import Gaussianize
from sklearn.preprocessing import StandardScaler
from dataset.create_dataset import *
from dataset.stock_price_crawler import get_stocks_price
def prepare_data(stock_index, end_date, days=7*365, batch_size=80, seq_len=127, use_dain = False):
    start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')
    df_original = get_stocks_price(stock_index, dataframe_type='wide', start_date=start_date, end_date=end_date)
    df = df_original.copy()
    df.columns = df.columns.droplevel(1)

    return_df = np.log(df['close']/df['close'].shift(1)).dropna()
    return_val = return_df.values.reshape(-1, 1)

    sc1 = StandardScaler()
    sc2 = StandardScaler()
    gaussianize = Gaussianize()
    if use_dain:
        log_returns_preprocessed = sc1.fit_transform(gaussianize.fit_transform(return_val))
    else:
        log_returns_preprocessed = sc2.fit_transform(gaussianize.fit_transform(sc1.fit_transform(return_val)))

    train_indices, val_indices = train_val_split(np.arange(len(log_returns_preprocessed)))
    train_data = np.array(log_returns_preprocessed[:val_indices[0]])
    val_data = np.array(log_returns_preprocessed[val_indices[0]:val_indices[-1]])

    train_dataset = PriceDataset(train_data, seq_len)
    val_dataset = PriceDataset(val_data, seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, return_val, log_returns_preprocessed, sc1, sc2, gaussianize
