import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import *
from modules import *
from critic import *
from actor import TSP_RL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
timer_batch  = Timer()
timer_epoch = Timer()
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def validate(data_loader, actor, reward_fn, plot_fn=None, save_dir='.',
             num_plot=5):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        static = static.to(device)
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)

        reward = reward_fn(static, tour_indices).mean().item()
        rewards.append(reward)

        if plot_fn is not None and batch_idx < num_plot:
            name = 'batch%d_%2.4f.png'%(batch_idx, reward)
            path = os.path.join(save_dir, name)
            plot_fn(static, tour_indices, path)

    actor.train()
    return np.mean(rewards)

def train(actor, critic, num_nodes, train_data, valid_data, reward_fn,
          plot_fn, num_epoches, batch_size, actor_lr, critic_lr, max_grad_norm,
          **kwargs):
    """
    Training function
    """
    save_dir = os.path.join('vrp', '%d' % num_nodes)

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    best_params = None
    best_reward = np.inf

    for epoch in range(num_epoches):
        actor.train()
        critic.train()

        losses, rewards, critic_rewards = [], [], []
        timer_batch.start()
        timer_epoch.start()
        for batch_idx, batch in enumerate(train_loader):

            static, dynamic, x0 = batch
            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None

            # Full forward pass through all features
            tour_indices, tour_logprob = actor(static, dynamic, x0)

            # Sum the log probabilities for each city in the tour
            reward = reward_fn(static, tour_indices)

            # Query the critic for an estimate of the reward
            critic_est = critic(static, dynamic).view(-1)

            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logprob.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * tour_logprob.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            # Backward pass for actor
            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            # Backward pass for critic
            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            losses.append(torch.mean(actor_loss.detach()).item())

            if (batch_idx + 1) % 100 == 0:
                # Stop the timer at the end of the epoch
                timer_batch.stop()

                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])

                print('Batch: [%d/%d] | reward: %2.3f | loss: %2.4f | elapsed time: %2.4fs' %
                      (batch_idx, len(train_loader), mean_reward, mean_loss,
                       timer_batch.times[-1]))
                # restart the time
                timer_batch.start()

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        # Save plotting of validation set tours
        valid_dir = os.path.join(save_dir, '%s' % epoch)

        mean_valid = validate(valid_loader, actor, reward_fn, plot_fn,
                              valid_dir, num_plot=5)

        # Save best model parameters
        if mean_valid < best_reward:
            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        timer_epoch.stop()
        print('Epoch: [%d/%d] | Mean epoch loss/reward: %2.4f, %2.4f, %2.4f | elapsed time: %2.4fs ' \
              '(%2.4fs / 100 batches)\n' % \
              (epoch, num_epoches, mean_loss, mean_reward, mean_valid, timer_epoch.times[-1],
               timer_epoch.avg()))

def train_vrp(args):

    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23  (Greedy)

    # Determines the maximum amount of load for a vehicle based on num nodes
    LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
    MAX_DEMAND = 9
    STATIC_SIZE = 2 # (x, y)
    DYNAMIC_SIZE = 2 # (load, demand)

    max_load = LOAD_DICT[args.num_nodes]

    train_data = VRPDataset(args.train_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       args.seed)

    valid_data = VRPDataset(args.valid_size,
                                       args.num_nodes,
                                       max_load,
                                       MAX_DEMAND,
                                       args.seed + 1)

    actor = TSP_RL(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args.hidden_size,
                    train_data._update_dynamic_feats,
                    train_data._update_mask,
                    args.num_layers,
                    args.dropout).to(device)

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = reward
    kwargs['plot_fn'] = plot_solution

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)

    test_data = VRPDataset(args.valid_size,
                                      args.num_nodes,
                                      max_load,
                                      MAX_DEMAND,
                                      args.seed + 2)

    test_dir = 'test'
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, reward, plot_solution, test_dir, num_plot=5)

    print('Average tour length: ', out)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='RL Vehicle Routing Problem')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--nodes', dest='num_nodes', default=20, type=int)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size',default=1000000, type=int)
    parser.add_argument('--valid-size', default=1000, type=int)

    args = parser.parse_args()

    train_vrp(args)

