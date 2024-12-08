import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from civ import Civilization
from pettingzoo.utils import wrappers
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import sys 
import matplotlib.pyplot as plt
# Import the PPO class and the Actor/Critic RNNs from your implementation
from train import ProximalPolicyOptimization, ActorRNN, CriticRNN

def preprocess_observation(obs):
    """
    Flatten the observation dictionary into a tensor.
    """
    # Flatten and concatenate all components of the observation
    map_obs = torch.tensor(obs['map'], dtype=torch.float32).flatten()
    units_obs = torch.tensor(obs['units'], dtype=torch.float32).flatten()
    cities_obs = torch.tensor(obs['cities'], dtype=torch.float32).flatten()
    money_obs = torch.tensor(obs['money'], dtype=torch.float32).flatten()
    obs_tensor = torch.cat([map_obs, units_obs, cities_obs, money_obs])
    return obs_tensor

def main():
    # Initialize the environment
    map_size = (15, 30)

    num_agents = 4
    sys.stdout.flush()
    env = Civilization(map_size, num_agents)
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)

    # Define hyperparameters
    hidden_size = 1024
    lambdaa = 0.01
    n_iters = 30
    n_fit_trajectories = 100
    n_sample_trajectories = 100
    max_steps = 50 #TO CHANGE
    num_epochs = 100

    # Initialize policies and optimizers
    actor_policies = {}
    critic_policies = {}
    theta_inits = {}
    env.reset()
    
    n_tiles = map_size[0]*map_size[1]

    for agent in env.agents:
        # Get observation and action spaces
        obs_space = env.observation_space(agent)
        act_space = env.action_space(agent)

        # Calculate input size for networks
        input_size = np.prod(obs_space['map'].shape) + \
                     np.prod(obs_space['units'].shape) + \
                     np.prod(obs_space['cities'].shape) + \
                     np.prod(obs_space['money'].shape)
        input_size_critic = np.prod(obs_space['map'].shape) + \
                     num_agents * np.prod(obs_space['units'].shape) + \
                     num_agents * np.prod(obs_space['cities'].shape) + \
                     num_agents * np.prod(obs_space['money'].shape)

        # Calculate output size for the actor (number of discrete actions)
        action_size = act_space['action_type'].n

        # Initialize actor and critic networks
        actor_policies[agent] = ActorRNN(input_size, hidden_size, n_tiles)
        critic_policies[agent] = CriticRNN(input_size_critic, hidden_size)

        # Initialize policy parameters
        theta_inits[agent] = np.random.randn(sum(p.numel() for p in actor_policies[agent].parameters()))

    # Instantiate PPO
    ppo = ProximalPolicyOptimization(
        env=env,
        actor_policies=actor_policies,
        critic_policies=critic_policies,
        lambdaa=lambdaa,
        theta_inits=theta_inits,
        n_iters=n_iters,
        n_fit_trajectories=n_fit_trajectories,
        n_sample_trajectories=n_sample_trajectories,
        max_steps = max_steps
    )
    def plot_rewards(reward_history, num_agents):
        plt.figure(figsize=(10, 6))
        for agent_idx in range(num_agents):
            plt.plot(reward_history[:, agent_idx], label=f"Agent {agent_idx}")
        plt.title("Total Rewards Across Training Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid()
        plt.show()

    # Train the policies
    ppo.train(eval_interval=30, eval_steps=100)
    print("trained")
    plot_rewards(ppo.reward_history, len(ppo.env.agents))
    sys.stdout.flush()

if __name__ == "__main__":
    main()