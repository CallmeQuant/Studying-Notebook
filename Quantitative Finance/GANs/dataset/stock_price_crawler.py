import pandas as pd
import requests
from pandas import json_normalize
from io import BytesIO
import time
from datetime import datetime, timedelta

entrade_headers = {
  'authority': 'services.entrade.com.vn',
  'accept': 'application/json, text/plain, */*',
  'accept-language': 'en-US,en;q=0.9',
  'dnt': '1',
  'origin': 'https://banggia.dnse.com.vn',
  'referer': 'https://banggia.dnse.com.vn/',
  'sec-ch-ua': '"Edge";v="114", "Chromium";v="114", "Not=A?Brand";v="24"',
  'sec-ch-ua-mobile': '?0',
  'sec-ch-ua-platform': '"Windows"',
  'sec-fetch-dest': 'empty',
  'sec-fetch-mode': 'cors',
  'sec-fetch-site': 'cross-site',
  'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1788.0'
}


def convert_date(text, data_type = "%Y-%m-%d"):
    return datetime.strptime(text, data_type)

def price_collector(symbol, start_date='2023-09-25', end_date='2023-10-25',
                    frequency='1D', type='stock', thousand_unit=True,
                    headers=entrade_headers):
    """
    Get historical price data from entrade.com.vn. The unit price is thousand VND.
    Parameters:
    ----------
        symbol (str): ticker of a stock or index. Available indices are: VNINDEX, VN30, HNX, HNX30, UPCOM, VNXALLSHARE, VN30F1M, VN30F2M, VN30F1Q, VN30F2Q
        start_date (str): start date of the historical price data
        end_date (str): end date of the historical price data
        frequency (str): frequency of the historical price data. Default is '1D' (daily), other options are '1' (1 minute), 15 (15 minutes), 30 (30 minutes), '1H' (hourly)
        type (str): stock or index. Default is 'stock'
        thousand_unit (bool): if True, convert open, high, low, close to VND for stock symbols. Default is True
        headers (dict): headers of the request
    Returns:
    pandas.DataFrame with the following format
        | time | open | high | low | close | volume |
        | ----------- | ---- | ---- | --- | ----- | ------ |
        | YYYY-mm-dd  | xxxx | xxxx | xxx | xxxxx | xxxxxx |
    """
    # add one more day to end_date
    end_date = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    # convert from_date, to_date to timestamp
    from_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    to_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
    url = f"https://services.entrade.com.vn/chart-api/v2/ohlcs/{type}?from={from_timestamp}&to={to_timestamp}&symbol={symbol}&resolution={frequency}"
    response = requests.request("GET", url, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        df = pd.DataFrame(response_data)
        df['t'] = pd.to_datetime(df['t'], unit='s') # convert timestamp to datetime
        df = df.rename(columns={'t': 'time', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}).drop(columns=['nextTime'])
        # add symbol column
        df['ticker'] = symbol
        df['time'] = df['time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Ho_Chi_Minh')
        # if frequency is 1D, then convert time to date
        if frequency == '1D':
            df['time'] = pd.to_datetime(df['time'].dt.date)
            df = df.rename(columns={'time': 'date'})
        else:
            pass
        # if type=stock and thousand_unit=False, then convert open, high, low, close to VND, elif type=index keep as is
        if type == 'stock' and thousand_unit == False:
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']] * 1000
            # convert open, high, low, close to int
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(int)
        else:
            pass
    else:
        print(f"Error in API response {response.text}", "\n")
    return df


def get_stocks_price(symbols, start_date='2023-06-25', end_date='2023-10-25',
                     frequency='1D', type='stock', thousand_unit=True,
                     dataframe_type = 'wide',
                     headers=entrade_headers):
    """
    Get historical price data from entrade.com.vn. The unit price is thousand VND.
    Parameters:
    ----------
        symbols (list): list of tickers of stocks or indices.
        start_date (str): start date of the historical price data
        end_date (str): end date of the historical price data
        frequency (str): frequency of the historical price data. Default is '1D' (daily), other options are '1' (1 minute), 15 (15 minutes), 30 (30 minutes), '1H' (hourly)
        type (str): stock or index. Default is 'stock'
        thousand_unit (bool): if True, convert open, high, low, close to VND for stock symbols. Default is True
        dataframe_type (str): long or wide. Default is 'wide'.
        headers (dict): headers of the request
    Returns:
    --------
    pandas.DataFrame
    """
    # Initialize a dictionary to store all stock dataframes
    all_stocks = {}

    # Loop over each symbol in the list
    for symbol in symbols:
        # Get the stock data for the current symbol
        df = price_collector(symbol, start_date, end_date, frequency, type, thousand_unit, headers)

        # Drop 'ticker' column and set 'date' column as index
        df = df.drop(columns=['ticker']).set_index('date')

        # Store the dataframe in the dictionary with its symbol as its key
        all_stocks[symbol] = df

    # Concatenate all dataframes in the dictionary
    if dataframe_type == 'long':
      all_stocks = pd.concat(all_stocks.values(), ignore_index=True)
    elif dataframe_type == 'wide':
      # Concatenate along columns and create multi-index from dictionary keys (symbols)
      all_stocks = pd.concat(all_stocks.values(), axis=1, keys=all_stocks.keys())
      all_stocks.fillna(0, inplace=True)

      # Swap levels in multi-index columns and sort them
      all_stocks.columns = all_stocks.columns.swaplevel(0, 1)
      all_stocks.sort_index(axis=1, level=[0, 1], inplace=True)

      # Rename column levels
      all_stocks.columns.names = ['Attributes', 'Tickers']

    return all_stocks

# Example
# Retrieve the Techcombank stock price
# end_date = '2023-10-25'
# start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days = 7 * 365)).strftime('%Y-%m-%d')
# df_original = get_stocks_price(['TCB'], dataframe_type = 'wide', start_date = start_date, end_date = end_date)
# print(df_original)

