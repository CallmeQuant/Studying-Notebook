import time
import torch
import numpy as np

def fix_seed(seed):
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