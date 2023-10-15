from model import mqrnn
from data_ultis import *
from scaler import *
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from plot import plot_data_forecast

# Define parameters dict, later add on input_size after reading the data
params_mqrnn = {
    'input_size': None,
    'num_layers': 3,
    'global_hidden_size': 64,
    'noise_hidden_size': 6,
    'global_num_layers': 3,
    'noise_num_layers': 1,
    'global_num_factors': 10,
    'lr': 1e-3,
    'prior': 168,
    'sequence_length': 60,
    'batch_size': 64,
    'num_epoches': 100,
    'iter_per_epoch': 3,
    'credible_interval': 95,
    'likelihood': 'continuous',
    'normalize': True,
    'max_scaler': False,
    'log_scaler': False,
    'mean_center': False,
    'probabilistic_CI': False
}

full_path = '/content/drive/MyDrive/data/sample_data.csv'
data, X, y, _, _ = load_and_process_data(full_path)

# print(X.shape)
# print(y.shape)
# Update params
params_mqrnn['input_size'] = X.shape[2]

# Initiate model
model = mqrnn.MQRNN(params_mqrnn)
num_ts, num_periods, num_feats = X.shape
optimizer = optim.Adam(model.parameters(), lr = params_mqrnn['lr'])
random.seed(28)

# Splitting dataset
X_train, y_train, X_test, y_test = split_train_test(X, y, split_ratio = 0.7)
losses = []
count = 0
scaler_target = None
if params_mqrnn['normalize']:
    scaler_target = Normalizer()
elif params_mqrnn['log_scaler']:
    scaler_target = LogScaler()
elif params_mqrnn['max_scaler']:
    scaler_target = MaxScaler()
elif params_mqrnn['mean_center']:
    scaler_target = MeanCenter()

if scaler_target is not None:
    y_train = scaler_target.fit_transform(y_train)

# training phase
progress = ProgressBar()
count = 0
for epoch in progress(range(params_mqrnn['num_epoches'])):
  for t in range(params_mqrnn['iter_per_epoch']):
    X_train_batch, y_train_batch, X_forecast, y_forecast = create_batch(X_train, y_train, params_mqrnn['prior'],
                                                                            params_mqrnn['sequence_length'], params_mqrnn['batch_size'])
    X_train_tensor, y_train_tensor = torch.from_numpy(X_train_batch).float(), torch.from_numpy(y_train_batch).float()
    X_forecast_tensor, y_forecast_tensor = torch.from_numpy(X_forecast).float(), torch.from_numpy(y_forecast).float()
    y_pred = model(X_train_tensor, y_train_tensor, X_forecast_tensor)

    # quantile loss
    loss = torch.zeros_like(y_forecast_tensor)
    num_ts = X_forecast_tensor.size(0)
    for q, rho in enumerate(params_mqrnn['quantiles']):
        y_pred_rho = y_pred[:, :, q].view(num_ts, -1)
        e = y_pred_rho - y_forecast_tensor
        loss += torch.max(rho * e, (rho - 1) * e)
    loss = loss.mean()

    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    count += 1

# Validation
Xtest = X_test[:, -params_mqrnn['sequence_length'] - params_mqrnn['prior']:-params_mqrnn['sequence_length'], :].reshape((num_ts, -1, num_feats))
Xforecast_test = X_test[:, -params_mqrnn['sequence_length']:, :].reshape((num_ts, -1, num_feats))
ytest = y_test[:, -params_mqrnn['sequence_length'] - params_mqrnn['prior']:-params_mqrnn['sequence_length']].reshape((num_ts, -1))
yforecast_test = y_test[:, -params_mqrnn['sequence_length']:].reshape((num_ts, -1))

if scaler_target is not None:
  ytest = scaler_target.transform(ytest)

y_pred = model(Xtest, ytest, Xforecast_test)
y_pred = y_pred.data.numpy()
if scaler_target is not None:
  y_pred = scaler_target.inverse_transform(y_pred)
y_pred = np.maximum(0, y_pred)

# Plot training loss
plt.plot(range(len(losses)), losses, "k-")
plt.xlabel("Period")
plt.ylabel("Loss")
plt.show()


# Plot forecasts
y_true_plot = np.hstack((scaler_target.inverse_transform(ytest), yforecast_test))
MQRNNfig = plot_data_forecast(params_mqrnn, y_true_plot, None, 'MQRNN', 'median', y_pred)