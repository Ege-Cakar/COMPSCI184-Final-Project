import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from civ import Civilization
from pettingzoo.utils import wrappers
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import wandb

# Import the PPO class and the Actor/Critic RNNs from your implementation
from train import ProximalPolicyOptimization
from rnn import ActorRNN, CriticRNN

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
    env = Civilization(map_size, num_agents)
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")

    # Define hyperparameters
    T = 500 #number of steps in trajectory
    batch_size = 20  # Define your batch size-- changed from 5
    K = 10 #number of minibatches to process
    lr = 5*0.0001
    hidden_size = 1024
    step_max = 100 #number of training iterations, has to be over 10 in order to save outputs
    # Initialize policies and optimizers
    actor_policies = {}
    critic_policies = {}
    theta_inits = {}
    env.reset()
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
        actor_policies[agent] = ActorRNN(input_size, hidden_size, action_size, env.max_cities, env.max_projects, device)
    critic_policies = CriticRNN(input_size_critic, hidden_size, device)
    
    sweep = False #TODO
    #hyperparameter sweep
    if sweep:
        wandb.login()
        def sweep_train(config):
            ppo = ProximalPolicyOptimization(
                env=env,
                actor_policies=actor_policies,
                critic_policies=critic_policies,
                step_max = 5,
                T = T, 
                batch_size = config.batch_size,
                K = config.K,
                lr = lr,
                device=device
            )
            return ppo.train()
      
        def sweep():
            wandb.init(project="hyperparameter-sweep-final", group="hyperparameter-sweep-group")
            score = sweep_train(wandb.config)
            print("FLAG")
            print(score)
            wandb.log({"score": score})


        sweep_configuration = {
            "method": "bayes",
            "metric": {"goal": "maximize", "name": "score"},
            "parameters": {
                "batch_size": {"values": [20, 25, 30, 35, 40]},
                "K": {"values": [10, 15, 20, 25, 30]},
                #"batch_size": {"min": 5, "max": 30},
                #"K": {"min": 5, "max": 30},
                #"lr": {"min": 0.00005, "max": 0.05}
            },
            "project": "hyperparameter-sweep",  # Project name in WandB
            "name": "my_sweep" 
        }
        
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="hyperparameter-sweep-final")
        wandb.agent("00jjuuyv", function=sweep, count=30)
    else:
        # Instantiate PPO
        ppo = ProximalPolicyOptimization(
            env=env,
            actor_policies=actor_policies,
            critic_policies=critic_policies,
            step_max=step_max,
            T = T, 
            batch_size = batch_size,
            K = K,
            lr = lr,
            device=device
        )
        # Train the policies
        ppo.train()


if __name__ == "__main__":
    main()