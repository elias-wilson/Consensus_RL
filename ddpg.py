import gc
import os
import torch

import torch.nn.functional as F
from torch.optim import Adam

from utils.nets import Actor, Critic

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG(object):

    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space, device, dtype, j, checkpoint_dir=None):
        """
        Deep Deterministic Policy Gradient
        Read the detail about it here:
        https://arxiv.org/abs/1509.02971

        Arguments:
            gamma:          Discount factor
            tau:            Update factor for the actor and the critic
            hidden_size:    Number of units in the hidden layers of the actor and critic. Must be of length 2.
            num_inputs:     Size of the input states
            action_space:   The action space of the used environment. Used to clip the actions and
                            to distinguish the number of outputs
            checkpoint_dir: Path as String to the directory to save the networks.
                            If None then "./saved_models/" will be used
        """

        self.j = j
        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space
        self.device = device
        self.dtype = dtype

        # Define the actor
        self.actor = Actor(hidden_size, num_inputs, action_space, device, dtype)
        self.actor_target = Actor(hidden_size, num_inputs, action_space, device, dtype)

        # Define the critic
        self.critic = Critic(hidden_size, num_inputs, action_space, device, dtype)
        self.critic_target = Critic(hidden_size, num_inputs, action_space, device, dtype)

        # Define the optimizers for both networks
        self.actor_optimizer = Adam(self.actor.parameters(),
                                    lr=1e-6)  # optimizer for the actor network
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=1e-3,
                                     weight_decay=1e-1
                                     )  # optimizer for the critic network

        # Make sure both targets are with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # Set the directory to save the models
        if checkpoint_dir is None:
            self.checkpoint_dir = "./saved_models/"
        else:
            self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def calc_action(self, state, action_noise=None):
        """
        Evaluates the action to perform in a given state

        Arguments:
            state:          State to perform the action on in the env.
                            Used to evaluate the action.
            action_noise:   If not None, the noise to apply on the evaluated action
        """

        if not torch.is_tensor(state):
            x = torch.tensor(state, device=self.device).type(self.dtype)
        else:
            x = state

        # Get the continous action value to perform in the env
        self.actor.eval()  # Sets the actor in evaluation mode
        mu = self.actor(x)
        self.actor.train()  # Sets the actor in training mode
        mu = mu.data

        # During training we add noise for exploration
        if action_noise is not None:
            noise = torch.Tensor(action_noise.noise()).to(self.device).type(self.dtype)
            mu += noise

        # Clip the output according to the action space of the env
        ones = torch.ones_like(mu,device=self.device,dtype=self.dtype)
        mu = torch.minimum(torch.maximum(mu,-ones),ones)

        return mu

    def calc_value(self, state, action):

        """
        Evaluate the predicted Q-value of a state-action pair

        """

        if not torch.is_tensor(state):
            x = torch.tensor(state, device=self.device).type(self.dtype)
        else:
            x = state
        if not torch.is_tensor(action):
            a = torch.tensor(action, device=self.device).type(self.dtype)
        else:
            a = action


        self.critic.eval()
        Q = self.critic(x, a)
        self.critic.train()
        return Q.data

    def update_params(self, batch):
        """
        Updates the parameters/networks of the agent according to the given batch.
        This means we ...
            1. Compute the targets
            2. Update the Q-function/critic by one step of gradient descent
            3. Update the policy/actor by one step of gradient ascent
            4. Update the target networks through a soft update

        Arguments:
            batch:  Batch to perform the training of the parameters
        """
        # Get tensors from the batch
        state_batch = batch['state']
        action_batch = batch['action']
        reward_batch = batch['reward']
        done_batch = batch['done']
        next_state_batch = batch['next_state']

        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_state_batch)

        # x = next_state_batch[:,4*self.j:4*(self.j+1)]
        # next_action_batch = torch.sigmoid(-15*x[:,[0,2]] - 5*x[:,[1,3]])
        next_state_action_values = self.critic_target(next_state_batch,
                                                    next_action_batch.detach())

        # Compute the target
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        expected_values = reward_batch + (1.0 - done_batch) * \
                                        self.gamma * next_state_action_values

        # TODO: Clipping the expected values here?
        # expected_value = torch.clamp(expected_value, min_value, max_value)

        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        policy_loss = self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def save_checkpoint(self, last_timestep):
        """
        Saving the networks and all parameters to a file in 'checkpoint_dir'

        Arguments:
            last_timestep:  Last timestep in training before saving
            replay_buffer:  Current replay buffer
        """
        checkpoint_name = self.checkpoint_dir + '/ep_{}.tar'.format(last_timestep)
        checkpoint = {
            'last_timestep': last_timestep,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_name)
        gc.collect()

    def get_path_of_latest_file(self):
        """
        Returns the latest created file in 'checkpoint_dir'
        """
        files = [file for file in os.listdir(self.checkpoint_dir) if \
                                (file.endswith(".pt") or file.endswith(".tar"))]
        filepaths = [os.path.join(self.checkpoint_dir, file) for file in files]
        last_file = max(filepaths, key=os.path.getctime)
        return os.path.abspath(last_file)

    def load_checkpoint(self, checkpoint_path=None):
        """
        Saving the networks and all parameters from a given path. If the given path is None
        then the latest saved file in 'checkpoint_dir' will be used.

        Arguments:
            checkpoint_path:    File to load the model from

        """

        if checkpoint_path is None:
            checkpoint_path = self.get_path_of_latest_file()

        if os.path.isfile(checkpoint_path):
            key = 'cpu' if self.device == 'cpu' else 'cuda'

            checkpoint = torch.load(checkpoint_path, map_location=key)
            start_timestep = checkpoint['last_timestep'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

            gc.collect()
            return start_timestep
        else:
            raise OSError('Checkpoint not found')

    def set_eval(self):
        """
        Sets the model in evaluation mode

        """
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def set_train(self):
        """
        Sets the model in training mode

        """
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

    def get_network(self, name):
        if name == 'Actor':
            return self.actor
        elif name == 'Critic':
            return self.critic
        else:
            raise NameError('name \'{}\' is not defined as a network'.format(name))
