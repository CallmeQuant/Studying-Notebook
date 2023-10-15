from model import deepfactors
from data_ultis import *
from loss_and_metrics import *
from scaler import *
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from plot import plot_data_forecast

# Define parameters dict, later add on input_size after reading the data
params_deepfactor = {
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
params_deepfactor['input_size'] = X.shape[2]

# Initiate model
model = deepfactors.DeepFactorRNN(params_deepfactor)
num_ts, num_periods, num_feats = X.shape
optimizer = optim.Adam(model.parameters(), lr = params_deepfactor['lr'])
random.seed(28)

# Splitting dataset
X_train, y_train, X_test, y_test = split_train_test(X, y, split_ratio = 0.7)
losses = []
count = 0
scaler_target = None
if params_deepfactor['normalize']:
    scaler_target = Normalizer()
elif params_deepfactor['log_scaler']:
    scaler_target = LogScaler()
elif params_deepfactor['max_scaler']:
    scaler_target = MaxScaler()
elif params_deepfactor['mean_center']:
    scaler_target = MeanCenter()

if scaler_target is not None:
    y_train = scaler_target.fit_transform(y_train)


progress = ProgressBar()
count = 0
for epoch in progress(range(params_deepfactor['num_epoches'])):
  for t in range(params_deepfactor['iter_per_epoch']):
    X_train_batch, y_train_batch, X_forecast, y_forecast = create_batch(X_train, y_train, params_deepfactor['prior'],
                                                                            params_deepfactor['sequence_length'], params_deepfactor['batch_size'])
    X_train_tensor, y_train_tensor = torch.from_numpy(X_train_batch).float(), torch.from_numpy(y_train_batch).float()
    X_forecast_tensor, y_forecast_tensor = torch.from_numpy(X_forecast).float(), torch.from_numpy(y_forecast).float()

    mu, sigma = model(X_train_tensor)
    print(mu.size())
    print(sigma.size())

    if params_deepfactor['likelihood'] == 'continuous':
      loss_c = Loss(y_train_tensor)
      loss = loss_c.gaussian_likelihood(mu, sigma)
    elif params_deepfactor['likelihood'] == 'count':
      loss_c = Loss(y_train_tensor)
      loss = loss_c.negative_binomial_likelihood(mu, sigma)

    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    count += 1

# Validation
Xtest = X_test[:, -params_deepfactor['sequence_length'] - params_deepfactor['prior']:-params_deepfactor['sequence_length'], :].reshape((num_ts, -1, num_feats))
Xforecast_test = X_test[:, -params_deepfactor['sequence_length']:, :].reshape((num_ts, -1, num_feats))
ytest = y_test[:, -params_deepfactor['sequence_length'] - params_deepfactor['prior']:-params_deepfactor['sequence_length']].reshape((num_ts, -1))
yforecast_test = y_test[:, -params_deepfactor['sequence_length']:].reshape((num_ts, -1))

if scaler_target is not None:
  ytest = scaler_target.transform(ytest)

result = []
n_samps = 100
for _ in tqdm(range(n_samps)):
  y_pred = model.sample(Xforecast_test)
  y_pred = y_pred.data.numpy()
  if scaler_target is not None:
    y_pred = scaler_target.inverse_transform(y_pred)
    # print(y_pred.shape)
  result.append(y_pred.reshape((-1,1))) # dim (num_periods, 1)

result = np.concatenate(result, axis=1) # dim (num_periods, num_samps)

# Evaluate
alpha = (100 - params_deepfactor['credible_interval']) / 2
median_forecast = np.median(result, axis=1)
result_deepfactor = evaluate(yforecast_test, median_forecast, result)
result_deepfactor['Model'] = 'Deep Factor'

# Plot training loss
plt.plot(range(len(losses)), losses, "k-")
plt.xlabel("Period")
plt.ylabel("Loss")
plt.show()

# Plot forecasts
y_true_plot = np.hstack((scaler_target.inverse_transform(ytest), yforecast_test))
Deepfactorfig = plot_data_forecast(params_deepfactor, y_true_plot, result, conditional_forecast = 'median')