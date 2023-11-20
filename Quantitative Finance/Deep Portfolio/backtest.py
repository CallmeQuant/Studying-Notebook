from ultis.backtest_ultis import *
from ultis.config import setup_model_config
from dataset.dataset import Dataset
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# Setting up config
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
config = {
  "MODEL": "GRU",
  "BATCH": 32,
  "SEED": 42,
  "EPOCHS": 300,
  "EARLY_STOP": 20,
  "LR": 0.005,
  "MOMENTUM": 0.9,
  "N_LAYER": 1,
  "HIDDEN_DIM": 128,
  "SEQ_LEN": 63,
  "DEVICE": device,
  "N_FEAT": 10,
  "DROPOUT": 0.2,
  "LEN_PRED": 21,
  "LEN_TRAIN": 63,
  "BIDIRECTIONAL": False,
  "USE_ATTENTION": True,
  "LB": 0,
  "UB": 0.2,
  "TCN": {
    "N_FEAT": 10,
    "N_OUT": 10,
    "KERNEL_SIZE": 4,
    "DROPOUT": 0.1,
    "SEQ_LEN": 63
  },
  "TRANSFORMER": {
    "N_FEAT": 10,
    "SEQ_LEN": 63,
    "N_LAYER": 6,
    "N_HEAD": 5,
    "DROPOUT": 0.1,
    "N_OUT": 10
  }
}
# Prepare data
stock_list = ['CII','DIG','HPG',"HT1",'HSG',"GAS","GVR",'TPB','TCB','MSN']
end_date = '2023-11-1'
data = Dataset(end_date, stock_list, 0.8, 63, 21, 32)
df_log_ret = data.get_data(end_date, stock_list=stock_list)
x_tr, y_tr, x_te, y_te, times_tr, times_te = data_split(
        df_log_ret,
        config['LEN_TRAIN'],
        config['lEN_PRED'],
        0.8,
        config['N_FEAT'],
    )

x_te_tensor = torch.from_numpy(x_te.astype("float32"))
checkpoint_path = '/Deep Portfolio/checkpoint'
model_list, model_names, _ = setup_model_config(config)
deep_weights, deep_portfolios = backtest_models(model_list, model_names, checkpoint_path,
                                 x_te_tensor, y_te, times_te, config, device)

# Backtesting
performance = pd.DataFrame(deep_portfolios).set_index('date')
result = compute_metrics(performance).sort_values(by=['Annualized Sharpe Ratio', 'Annualized Return'], ascending = False)
print(result)

performance_melted = performance.reset_index().melt('date', var_name='portfolio', value_name='wealth')
# Create plot
plt.figure(figsize=(10,5))
sns.lineplot(data=performance_melted, x='date', y='wealth', hue='portfolio')
plt.title('Wealth of each portfolio over backtest period')
plt.legend(title='Portfolio')
plt.show()