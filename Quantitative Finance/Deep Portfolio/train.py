import torch
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import math
# Optimizer
from modules.optimizer import SAM
# Loss function
from modules.loss import max_sharpe
# dataset
from dataset.dataset import Dataset
# Training ultis
from ultis.train_ultis import fix_seed, Timer, get_cosine_lr_scheduler
from ultis.save_load import save_model
from ultis.config import setup_model_config
# Plotting
import matplotlib.pyplot as plt
plot_params = {
    "font.size": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": False,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.titlesize": 15,
    "figure.dpi": 150,
    "figure.constrained_layout.use": True,
    "figure.autolayout": False}
plt.rcParams.update(plot_params)

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
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Function for setting the seed
fix_seed(config['SEED'])

# Setting up models
model_names, model_list, model_dict = setup_model_config(config)

# Data retrieving
stock_list = ['CII','DIG','HPG',"HT1",'HSG',"GAS","GVR",'TPB','TCB','MSN']
end_date = '2023-11-1'
# start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days = 6 * 365 - 1)).strftime('%Y-%m-%d')
data = Dataset(end_date, stock_list, 0.8, 63, 21, 32)
df_log_ret = data.get_data(end_date, stock_list = stock_list)
train_loader = data.get(train_mode = True)
test_loader = data.get(train_mode = False)

checkpoint_path = '/Deep Portfolio/checkpoint'

def train_model(model_name, constraint_allocation = True):
  timer = Timer()
  model = model_dict[model_name]
  base_optimizer = torch.optim.SGD
  optimizer = SAM(
  model.parameters(), base_optimizer, lr=config["LR"]
  , momentum=config['MOMENTUM']
  )

  valid_loss = []
  train_loss = []

  early_stop_count = 0
  early_stop_th = config["EARLY_STOP"]

  criterion = max_sharpe
  lr_scheduler = get_cosine_lr_scheduler(4e-3, 1e-6)

  for epoch in tqdm(range(config["EPOCHS"])):
      timer.start()
      # Update learning rate
      lr = lr_scheduler(config["EPOCHS"], epoch)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr

      for phase in ["train", "valid"]:
          if phase == "train":
              model.train()
              dataloader = train_loader
          else:
              model.eval()
              dataloader = test_loader

          batch_loss = 0.0

          for idx, data in enumerate(dataloader):
              x, y = data
              x = x.to(device)
              y = y.to(device)
              optimizer.zero_grad()
              with torch.set_grad_enabled(phase == "train"):
                  out = model(x)
                  if constraint_allocation:
                    loss = criterion(y, out)
                  else:
                    loss = criterion(y, out, regularization=False)
                  if phase == "train":
                      loss.backward()
                      optimizer.first_step(zero_grad=True)
                      if constraint_allocation:
                        criterion(y, model(x)).backward()
                      else:
                        criterion(y, out, regularization=False)
                      optimizer.second_step(zero_grad=True)

              batch_loss += loss.item() / len(dataloader)
          if phase == "train":
            train_loss.append(batch_loss)
          else:
            valid_loss.append(batch_loss)
            if batch_loss <= min(valid_loss):
                save_model(model, model_path = checkpoint_path, model_name = model_name)
                print(f"Epoch {epoch + 1} : loss enhanced with {batch_loss}")
                early_stop_count = 0
            else:
                early_stop_count += 1


      if early_stop_count == early_stop_th:
        break
      # Stop the timer at the end of the epoch
      timer.stop()
      if (epoch % 10 == 0):
        print(f'Epoch: [{epoch+1}/{config["EPOCHS"]}]'
              + f' | Training loss: {train_loss[-1]:.3f}'
              + f' | Validiation loss: {valid_loss[-1]:.3f}'
              + f' | Time per epoch: {timer.times[-1]:.2f} seconds')

  total_training_time = timer.sum()
  print(f'Total training times: {total_training_time/60:.2f} minutes')
  return model, train_loss, valid_loss, total_training_time

# Training session
runtime_all = {}
visualize = True
rows = math.ceil(len(model_names)/2.0)
fig, axes = plt.subplots(rows, 2, figsize = (10, 5))

for idx, name in enumerate(model_names):
  print(f'Starting training model: {name}')
  print("-"*50)
  model, train_loss, valid_loss, runtime = train_model(name, constraint_allocation=True)
  runtime_all[name] = runtime

  if visualize:
    ax = axes.flatten()[idx]
    ax.plot(range(1, len(train_loss)+1), train_loss, label='Train Loss')
    ax.plot(range(1, len(valid_loss)+1), valid_loss, label='Valid Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title(name)
    ax.legend()
    if idx == len(model_names):
      fig.suptitle('Train and valid loss for each model')
    plt.show()

# Plotting runtime
# Convert the dictionary to a DataFrame
df_runtime = pd.DataFrame(list(runtime_all.items()), columns=['Models', 'Time'])

# Create the barplot
plt.figure(figsize=(10,5))
sns.barplot(x='Models', y='Time', data=df_runtime, color='tab:orange')

# Set labels and title
plt.xlabel('Models')
plt.ylabel('Time (seconds)')
plt.title('Model Running Times')
plt.show()