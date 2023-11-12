import os
import torch.optim as optim
from tqdm import tqdm
from dataset.create_dataset import *
from dataset.data_preparation import prepare_data
from modules.model import Discriminator, Generator
from modules.toolkits_train import fix_seed, _retrieve_model_ct, Timer, get_cosine_lr_scheduler, save_model
from transform.acf import rolling_window, acf
import matplotlib.pyplot as plt

# Setting up plotting params
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

## Setting up for training
fix_seed(42)
# Check if cuda is available
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# path for storing model
checkpoint_path = 'Quantitative Finance/GANs/Generator'
# Hyperparameters
num_epochs = 4
weight_decay = 1e-4
nz = 3
batch_size = 80
seq_len = 127
clip= 0.01
lr = 0.0002
# Bookkeeping variables and Time measure object
timer = Timer()
loss = {'D': [],
        'G': []}

lr_scheduler = get_cosine_lr_scheduler(1e-3, 1e-6)

## Downloading data
end_date = '2023-10-25'
stock_index = ['VNM']

train_dataloader, val_dataloader, return_val, log_returns_preprocessed_og, sc1,\
  sc2, gaussianize = prepare_data(stock_index, end_date)


def train_model(log_ret, hidden_noise_size, num_epochs,
                batch_size, seq_len, weight_decay, lr, clip,
                train_mode = True,
                use_last_epoch = True,
                use_cuda = use_cuda,
                device = device,
                required_checkpoint = None):
  model_name = 'checkpoint_1'
  dataset = PriceDataset(log_ret, seq_len)
  dataloader = DataLoader(dataset, batch_size=batch_size)
  discriminator = Discriminator(seq_len)
  generator = Generator()
  if use_cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
  optimizerD = optim.Adam(discriminator.parameters(), weight_decay = weight_decay, lr=lr)
  optimizerG = optim.Adam(generator.parameters(), weight_decay = weight_decay, lr=lr)
  pretrained_exists = os.path.isfile(_retrieve_model_ct(checkpoint_path, model_name))

  if pretrained_exists and train_mode:
    print('Checkpoint exists, but will be overwritten')
    for epoch in tqdm(range(num_epochs)):
      # start timer
      timer.start()

      sample_loss_D = 0.
      sample_loss_G = 0.

      # Update learning rate
      lr = lr_scheduler(num_epochs, epoch)
      for param_group in optimizerD.param_groups:
          param_group['lr'] = lr
      for param_group in optimizerG.param_groups:
          param_group['lr'] = lr

      for batch_idx, data in enumerate(train_dataloader, 0):

        discriminator.zero_grad()  # Zero the gradients
        real = data.to(device)
        batch_size, seq_len = real.size(0), real.size(2)
        noise = torch.randn(batch_size, hidden_noise_size, seq_len, device=device)
        fake = generator(noise).detach()

        loss_D = -torch.mean(discriminator(real)) + torch.mean(discriminator(fake))  # Compute the loss
        loss_D.backward()  # Backward pass
        optimizerD.step()  # Update the weights
        sample_loss_D += loss_D.item()
        for param in discriminator.parameters():
          param.data.clamp_(-clip, clip)
        if batch_idx % 5 == 0:
          generator.zero_grad()  # Zero the gradients
          loss_G = -torch.mean(discriminator(generator(noise)))  # Compute the loss
          loss_G.backward()  # Backward pass
          optimizerG.step()  # Update the weights
          sample_loss_G += loss_G.item()

      loss_D_at_epoch = sample_loss_D / len(dataloader)
      loss_G_at_epoch = sample_loss_G / len(dataloader)

      loss['D'].append(loss_D_at_epoch)
      loss['G'].append(loss_G_at_epoch)

      # Stop the timer at the end of the epoch
      timer.stop()
      print(
          f'Epoch: [{epoch}/{num_epochs}]'
          + f' | Discriminator loss: {loss_D_at_epoch:.3f}'
          + f' | Generator loss: {loss_G_at_epoch:.3f}'
          + f' | Time per epoch: {timer.times[-1]:.2f} seconds')

      try:
          os.makedirs(checkpoint_path, exist_ok = True)
      except OSError as error:
          print("Directory can not be created")
      # checkpoint_generator = {'epoch_idx': epoch,
      #                               'model_state_dict': generator.state_dict(),
      #                               'optimizer_state_dict': optimizerG.state_dict(),}
      # checkpoint_file = os.path.join(checkpoint_path, f'checkpoint_{epoch}.pth')
      # torch.save(checkpoint_generator, checkpoint_file)
      save_model(epoch, generator, optimizerG, checkpoint_path)
  else:
      checkpoint_files = [os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path)
                              if f.startswith('checkpoint_')]
      if use_last_epoch:
        checkpoint = torch.load(checkpoint_files[-1])
      else:
        if required_checkpoint is None:
          raise ValueError('Can not load checkpoint model if not provide specific epoch index')
        required_checkpoint_filename = 'checkpoint_' + str(required_checkpoint)
        for filename in checkpoint_files:
          if filename == required_checkpoint_filename:
            checkpoint = torch.load(filename)

      state_dict = checkpoint['model_state_dict']
      generator.load_state_dict(state_dict)


train_model(log_returns_preprocessed_og, nz, num_epochs,
                batch_size, seq_len, weight_decay, lr, clip)

total_time_all_epoch = timer.sum()
print("")
print(f"Total training time: {total_time_all_epoch/60:.2f} minutes")

fig, ax1 = plt.subplots(1, 1, figsize = (10, 5))
ax1.plot(range(1, num_epochs + 1), loss['G'], label='Generator Loss', marker='o')
ax1.plot(range(1, num_epochs + 1), loss['D'], label='Discriminator Loss', marker='o')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_title('Generator and Discriminator Loss')
ax1.legend()

generator = train_model(train_mode=False)
scores = {}
for batch_idx, data in enumerate(val_dataloader, 0):
  scores[batch_idx] = []
  real = data.to(device)
  batch_size, seq_len = real.size(0), real.size(2)
  real = real.cpu().detach().squeeze(1).permute(1, 0)

  acf_real = acf(real, batch_size).mean(axis = 1, keepdims = True)
  abs_acf_real = acf(real**2, batch_size).mean(axis = 1, keepdims = True)
  le_real = acf(real, batch_size, le=True).mean(axis = 1, keepdims = True)

  noise = torch.randn(batch_size, nz, seq_len, device=device)
  y = generator(noise).cpu().detach()
  acf_fake = acf(y.squeeze(1).permute(1, 0), batch_size).mean(axis = 1, keepdims = True)
  abs_acf_fake = acf((y.squeeze(1).permute(1, 0))**2, batch_size).mean(axis = 1, keepdims = True)
  le_fake = acf(y.squeeze(1).permute(1, 0), batch_size, le = True).mean(axis = 1, keepdims = True)

  scores[batch_idx].append(np.linalg.norm(acf_real - acf_fake))
  scores[batch_idx].append(np.linalg.norm(abs_acf_real - abs_acf_fake))
  scores[batch_idx].append(np.linalg.norm(le_real - le_fake))

  print(f'Batch id: [{batch_idx+1}/{len(val_dataloader)}]'
        + f' | ACF score: {scores[batch_idx][0]:.3f}'
        + f' | Abs ACF score: {scores[batch_idx][1]:.3f}'
        + f' | Leverage effect score: {scores[batch_idx][2]:.3f}')

generator.eval()
noise = torch.randn(80, 3, 127).to(device)
y = generator(noise).cpu().detach().squeeze()

y = (y - y.mean(axis=0)) / y.std(axis=0)
y = sc2.inverse_transform(y)
y = np.array([gaussianize.inverse_transform(np.expand_dims(x, 1)) for x in y]).squeeze()
y = sc1.inverse_transform(y)

# # some basic filtering to redue the tendency of GAN to produce extreme returns
y = y[(y.max(axis=1) <= 2 * return_val.max()) & (y.min(axis=1) >= 2 * return_val.min())]
y -= y.mean()


## Plotting (run it to check distribution)
# fig, ax = plt.subplots(figsize=(16,9))
# ax.plot(np.cumsum(y[0:20], axis=1).T, alpha=0.75)
# ax.set_title('20 generated log return paths'.format(len(y)))
# ax.set_xlabel('days')
# ax.set_ylabel('Cumalative log return')
#
# n_bins = 50
# windows = [1, 10, 20, 30]
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
# for i, ax in enumerate(axes.flatten()):
#     real_dist = rolling_window(return_val, windows[i], sparse = not (windows[i] == 1)).sum(axis=0).ravel()
#     fake_dist = rolling_window(y.T, windows[i], sparse = not (windows[i] == 1)).sum(axis=0).ravel()
#     ax.hist([real_dist, fake_dist], bins=50, density=True)
#     ax.set_xlim(*np.quantile(fake_dist, [0.001, .999]))
#
#     ax.set_title(f'{windows[i]} day return distribution')
#     ax.yaxis.grid(True, alpha=0.5)
#     ax.set_xlabel('Cumulative log return')
#     ax.set_ylabel('Frequency')
#     ax.legend(['Historical returns', 'Synthetic returns'])