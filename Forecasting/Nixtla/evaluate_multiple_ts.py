# A function to compute the accuracy of forecasts using different accuracy metrics using Nixtla framework. 
# This function will return a multiindex dataframe with level 0 and level 1 being the forecasting methods and the timeseries' indices respectively. 
# The dataframe input must match the format of Nixtla, which requires a id column named 'unique_id' and a date column named 'ds'.

# Reference: https://github.com/Nixtla/datasetsforecast

from datasetsforecast.losses import mse, mae, smape, mape
from typing import List, Callable

metrics = [mse, mae, smape, mape]

def evaluate(df: pd.DataFrame, metrics: List[Callable]) -> pd.DataFrame:
    id_ts = list(df['unique_id'].unique())
    eval_ = {}
    models = df.loc[:, ~df.columns.str.contains('unique_id|y|ds|cutoff|lo|hi')].columns
    for model in models:
        eval_[model] = {}
        for id in id_ts:
          metric_dict = {}
          for metric in metrics:
            metric_name = metric.__name__
            # res = df.loc[:, df.columns.str.contains('unique_id|y|ds|' + str(model))].groupby("unique_id").apply(lambda x: metric(x['y'], x[model])).to_frame(metric.__name__)
            y_hat = df.loc[df['unique_id'] == id, df.columns.str.contains(str(model))].values.flatten()
            # print(y_hat)
            y = df.loc[df['unique_id'] == id, df.columns.str.contains('y')].values.flatten()
            # print(y)
            metric_dict[metric_name] = metric(y, y_hat)
        
          eval_[model][id] = metric_dict
    
    model_id = []
    id_ts = []
    for model, ts in eval_.items():
      model_id.append(model)
      id_ts.append(pd.DataFrame.from_dict(ts, orient='index'))
    
    eval_df = pd.concat(id_ts, keys = model_id)
    eval_df.columns = eval_df.columns.str.upper()
    return eval_df, eval_
