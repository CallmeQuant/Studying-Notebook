import numpy as np
import torch
import time
import os
import json

def fix_seed(seed):
    """Function for setting seed"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class Timer:
  """Record multiple running times."""
  def __init__(self):
    """Defined in :numref:`sec_minibatch_sgd`"""
    self.times = []
    self.start()

  def start(self):
    """Start the timer."""
    self.tik = time.time()

  def stop(self):
    """Stop the timer and record the time in a list."""
    self.times.append(time.time() - self.tik)
    return self.times[-1]

  def avg(self):
    """Return the average time."""
    return sum(self.times) / len(self.times)

  def sum(self):
    """Return the sum of time."""
    return sum(self.times)

  def cumsum(self):
    """Return the accumulated time."""
    return np.array(self.times).cumsum().tolist()

def _retrieve_model_ct(model_path, model_name):
    """Get the model file that stores network parameters"""
    return os.path.join(model_path, model_name + ".pth")

def _retrieve_model_config(model_path, model_name):
    """Get the model file that stores configurations/hyperparameters"""
    return os.path.join(model_path, model_name + ".json")

def load_model(model_path, model_name, network, device):
    """Load saved models from path"""
    config_file, model_file = _retrieve_model_config(model_path, model_name), \
    _retrieve_model_ct(model_path, model_name)
    assert os.path.isfile(config_file), f"Config file \"{config_file}\" is not found. Check the path again"
    assert os.path.isfile(model_file), f"Model file \"{model_file}\" is not found. Check the path again"
    with open(config_file, "r") as f:
        config_dict = json.load(config_file)

    network.load_state_dict(torch.load(model_file, map_location=device))
    return network

def save_model(epoch, model, optimizer, path):
  checkpoint_generator = {'epoch_idx': epoch,
                                  'model_state_dict': model.state_dict(),
                                  'optimizer_state_dict': optimizer.Chastate_dict(),}
  checkpoint_file = os.path.join(path, f'checkpoint_{epoch}.pth')
  torch.save(checkpoint_generator, checkpoint_file)

def get_cosine_lr_scheduler(init_lr, final_lr):
    def lr_scheduler(n_epoch, epoch_idx):
        lr = final_lr + 0.5 * (init_lr - final_lr) * (1 + np.cos(np.pi * epoch_idx / n_epoch))
        return lr

    return lr_scheduler


def get_multiplicative_lr_scheduler(init_lr, drop_at, multiplicative_factor):
    def lr_scheduler(n_epoch, epoch_idx):
        lr = init_lr
        for epoch in drop_at:
            if epoch_idx + 1 >= epoch:
                lr *= multiplicative_factor
        return lr

    return lr_scheduler