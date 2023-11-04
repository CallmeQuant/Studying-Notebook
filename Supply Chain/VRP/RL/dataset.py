import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Timer:
  """
  Record multiple running times.
  """
  def __init__(self):
    """
    Defined in :numref:`sec_minibatch_sgd`
    """
    self.times = []
    self.start()

  def start(self):
    """
    Start the timer.
    """
    self.tik = time.time()

  def stop(self):
    """
    Stop the timer and record the time in a list.
    """
    self.times.append(time.time() - self.tik)
    return self.times[-1]

  def avg(self):
    """
    Return the average time.
    """
    return sum(self.times) / len(self.times)

  def sum(self):
    """
    Return the sum of time.
    """
    return sum(self.times)

  def cumsum(self):
    """
    Return the accumulated time.
    """
    return np.array(self.times).cumsum().tolist()

def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class VRPDataset(Dataset):
    """
    Dataset class for Vehicle Routing Problem
    """
    def __init__(self, num_samples, input_size, max_load = 20, max_demand = 9, seed = None):
        super(VRPDataset, self).__init__()

        if max_load < max_demand:
            raise ValueError('Maximum demand of customers can not exceed the maximum load of vehicles')

        if seed is None:
            seed = np.random.randint(1234567890)

        fix_seed(seed)

        self.num_samples = num_samples
        self.max_load = max_load
        self.max_demand = max_demand

        # Including the location of depot, which is 0
        # shape: num_samples x (x-coord, y-coord), num_customers + 1)
        locations_coord = torch.rand((num_samples, 2, input_size + 1))
        self.static_feats = locations_coord

        # Dynamic feats
        # trucks' loads
        # To avoid passing large values to neural network => scale values to [0,1]
        dynamic_shape = (num_samples, 1, input_size + 1)
        loads = torch.full(dynamic_shape, 1.)

        # customers' demands
        # Same as loads, to avoid large number => scale values by maximum loads
        demands = torch.randint(1, max_demand + 1, dynamic_shape)
        demands = demands / float(max_load)

        # Set demands at depot to 0
        demands[:, 0, 0] = 0
        self.dynamic_feats = torch.tensor(np.concatenate((loads, demands), axis = 1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # (static_feats, dynamic_feats, start_location)
        return (self.static_feats[index], self.dynamic_feats[index], self.static_feats[index, :, 0:1])

    def _update_mask(self, mask, dynamic_feats, chosen_index = None):
        """
        Updates the mask to hide non-valid states

        Parameters
        ----------
        dynamic: torch.autograd.Variable (num_samples, num_feats, seq_len)
        """

        loads = dynamic_feats.data[:, 0] # (batch_size, seq_len)
        demands = dynamic_feats.data[:, 1]  # (batch_size, seq_len)

        # Masking procedures
        # Customers with zero demands can not be visited => If no positive demand, end tour
        if demands.eq(0).all():
            return demands * 0

        # feasible states of mask: customers with positive demands larger than loads will be masked
        updated_mask = demands.ne(0) * demands.lt(loads)

        # Avoiding going to depot back and forth
        return_home = chosen_index.ne(0)

        # If vehicle just visited depot, allow to visit again. Otherwise, not allowed
        if return_home.any():
            updated_mask[return_home.nonzero(), 0] = 1.

        if (~return_home).any():
            updated_mask[(~return_home).nonzero(), 0] = 0.

        no_load_remain = loads[:, 0].eq(0).float()
        no_demands_remain = demands[:, 1:].sum(1).eq(0).float()

        merged = (no_load_remain + no_demands_remain).gt(0)

        if merged.any():
            updated_mask[merged.nonzero(), 0] = 1.
            updated_mask[merged.nonzero(), 1:] = 0.

        return updated_mask.float()

    def _update_dynamic_feats(self, dynamic_feats, chosen_index):
        """
        Updates the loads and demands dataset
        """
        # Update the dynamic elements differently for if we visit depot vs. a city
        visit = chosen_index.ne(0)
        depot = chosen_index.eq(0)

        # Clone dynamic feats
        all_loads = dynamic_feats[:, 0].clone()
        all_demands = dynamic_feats[:, 1].clone()

        # Extract load and demand corresponding to vehicle/customer
        load = torch.gather(all_loads, 1, chosen_index.unsqueeze(1))
        demand = torch.gather(all_demands, 1, chosen_index.unsqueeze(1))

        # Along all minibatch, if we choose a customer, we will satisfy demands as much as possible
        if visit.any():
            new_load = torch.clamp(load - demand, min = 0)
            new_demand = torch.clamp(demand - load, min = 0)

            # Squeeze to be a number
            visit_index = visit.nonzero().squeeze()

            all_loads[visit_index] = new_load[visit_index]
            all_demands[visit_index, chosen_index[visit_index]] = new_demand[visit_index].view(-1)
            # Check how much demand/capacity is used up at depot when vehicles take on new load
            all_demands[visit_index, 0] = -1. + new_load[visit_index].view(-1)

        # Back to depot to refill
        if depot.any():
            all_loads[depot.nonzero().squeeze()] = 1.
            all_demands[depot.nonzero().squeeze(), 0] = 0.

        updated_dynamic_feats = torch.cat((all_loads.unsqueeze(1), all_demands.unsqueeze(1)), 1)
        return torch.tensor(updated_dynamic_feats.data, device = dynamic_feats.device)

def reward(static_feats, tour_indices):
    """
    Given tour indices, compute the Euclidean distance between all customers' locations (cities)
    """

    # # Convert the indices back into a tour
    index = tour_indices.unsqueeze(1).expand(-1, static_feats.size(1), -1)
    tour = torch.gather(static_feats.data, 2, index).permute(0, 2, 1)

    # Add the depot location
    start = static_feats.data[:, :, 0].unsqueeze(1)
    y = torch.cat((start, tour, start), dim = 1)

    # Euclidean distance between two consecutive points
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    return tour_len.sum(1)

def plot_solution(static, tour_indices, save_path):
    """Plots the found solution."""

    plt.close('all')

    num_plots = 3 if int(np.sqrt(len(tour_indices))) >= 3 else 1

    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')

    if num_plots == 1:
        axes = [[axes]]
    axes = [a for ax in axes for a in ax]

    for i, ax in enumerate(axes):

        # Convert the indices back into a tour
        idx = tour_indices[i]
        if len(idx.size()) == 1:
            idx = idx.unsqueeze(0)

        idx = idx.expand(static.size(1), -1)
        data = torch.gather(static[i].data, 1, idx).cpu().numpy()

        start = static[i, :, 0].cpu().data.numpy()
        x = np.hstack((start[0], data[0], start[0]))
        y = np.hstack((start[1], data[1], start[1]))

        # Assign each subtour a different colour & label in order traveled
        idx = np.hstack((0, tour_indices[i].cpu().numpy().flatten(), 0))
        where = np.where(idx == 0)[0]

        for j in range(len(where) - 1):

            low = where[j]
            high = where[j + 1]

            if low + 1 == high:
                continue

            ax.plot(x[low: high + 1], y[low: high + 1], zorder=1, label=j)

        ax.legend(loc="upper right", fontsize=3, framealpha=0.5)
        ax.scatter(x, y, s=4, c='r', zorder=2)
        ax.scatter(x[0], y[0], s=20, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)


# Example - Reward function
# static_feats = torch.tensor([
#     [[0, 1, 2], [0, 1, 2]],  # Sample 1: coordinates of depot, city1, city2
#     [[0, -1, -2], [0, -1, -2]]  # Sample 2: coordinates of depot, city1, city2
# ])
#
# tour_indices = torch.tensor([
#     [0, 1, 2],  # Tour for sample 1
#     [0, 2, 1]   # Tour for sample 2
# ])
#
# idx = tour_indices.unsqueeze(1).expand(-1, static_feats.size(1), -1)
# tour = torch.gather(static_feats.data, 2, idx).permute(0, 2, 1)
# start = static_feats.data[:, :, 0].unsqueeze(1)
# y = torch.cat((start, tour, start), dim=1)
# tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

# Example
# # Define a VRPDataset object
# vrp_dataset = VRPDataset(num_samples=1, input_size=5)
#
# # Get a sample from the dataset
# static_feats, dynamic_feats, start_location = vrp_dataset[0]
#
# # Print static features (locations)
# print("Static features:")
# print(static_feats)
#
# # Print dynamic features (loads and demands)
# print("Dynamic features:")
# print(dynamic_feats)

# Initialize a mask with all ones
# mask = torch.ones((1, 6))
#
# # Print initial mask
# print("Initial mask:")
# print(mask)
#
# # Choose a customer index (for example, 3)
# chosen_index = torch.tensor([3])
#
# # Update mask using update_mask method
# mask = vrp_dataset._update_mask(mask, dynamic_feats.unsqueeze(0), chosen_index)
#
# # Print updated mask
# print("Updated mask:")
# print(mask)


        



