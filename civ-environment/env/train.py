import numpy as np
import pettingzoo as pz
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import AECEnv
import pygame
from pygame.locals import QUIT
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from pettingzoo.utils import wrappers
import scipy.optimize as opt
from typing import TypeVar, Callable
import jax.numpy as jnp
import torch.nn.functional as F
from tqdm import tqdm
import sys

from civ import Civilization

# Type aliases for readability
State = TypeVar("State")  # Represents the state type
Action = TypeVar("Action")  # Represents the action type

class ProximalPolicyOptimization:
    def __init__(self, env, actor_policies, critic_policies, lambdaa, theta_inits, n_iters, n_fit_trajectories, n_sample_trajectories, max_steps):
        """
        Initialize PPO with environment and hyperparameters.

        Args:
            env: The environment for training.
            pi: Policy function that maps parameters to a function defining action probabilities.
            lambdaa: Regularization coefficient.
            theta_init: Initial policy parameters.
            n_iters: Number of training iterations.
            n_fit_trajectories: Number of trajectories for fitting the advantage function.
            n_sample_trajectories: Number of trajectories for optimizing the policy.
            max_steps: Number of iterations to run trajectory for
        """
        self.env = env
        self.actor_policies = actor_policies
        self.critic_policies = critic_policies 
        self.lambdaa = lambdaa
        self.theta_inits = theta_inits
        self.n_iters = n_iters
        self.n_fit_trajectories = n_fit_trajectories
        self.n_sample_trajectories = n_sample_trajectories
        self.max_steps = max_steps
        self.reward_history = np.zeros((self.n_iters, len(self.env.agents)))

    def train(self, eval_interval, eval_steps):
        """
        Proximal Policy Optimization (PPO) implementation.

        Args:
            env: The environment in which the agent will act.
            actor_policies: RNN defined in test.py
            critic_policies: RNN defined in test.py
            λ: Regularization coefficient to penalize large changes in the policy.
            theta_inits: Initial parameters of the policy, conists of k1 through k10 and epsilon (environmental impact)
            n_iters: Number of training iterations.
            n_fit_trajectories: Number of trajectories to collect for fitting the advantage function.
            n_sample_trajectories: Number of trajectories to collect for optimizing the policy.
        
        Returns:
            Trained policy parameters, theta.
        """
        actor_optimizers = {
            agent: torch.optim.Adam(policy.parameters(), lr=1e-3)
            for agent, policy in self.actor_policies.items()
        }

        critic_optimizers = {
            agent: torch.optim.Adam(policy.parameters(), lr=1e-3)
            for agent, policy in self.critic_policies.items()
        }
        #This code is taken from class
        theta_list = self.theta_inits

        for iter in range(self.n_iters):
            print("Training iteration:" ,iter)
            sys.stdout.flush()
            n_agents = len(self.actor_policies)
            self.env.reset()   
            
            trajectories = self.initialize_starting_trajectories(self.env, self.actor_policies, n_agents)


            for step in range(self.max_steps):

                for agent in self.env.agents:

                    sys.stdout.flush()               
                    trajectories_next_step = self.sample_trajectories( #for a single agent
                        self.env,
                        self.actor_policies[agent],
                        self.critic_policies[agent],
                        trajectories[agent],
                        n_agents,
                        iter,
                        agent
                    )
                trajectories[agent].extend(trajectories_next_step)


            # Process trajectories
            states = torch.stack([t[0] for t in trajectories])  # States
            obs = torch.stack([self.flatten_observation(t[1]) for t in trajectories])
            actions = torch.stack([t[4] for t in trajectories])  # Actions
            rewards = torch.tensor([t[5] for t in trajectories], dtype=torch.float32)  # Rewards
            next_states = torch.stack([t[6] for t in trajectories])  # Next states
            

            # Compute returns (discounted rewards-to-go)
            returns = self.compute_returns(rewards)

            # Compute values from critic
            values = []
            for agent, critic_policy in self.critic_policies.items():
                agent_states = states[agent]  # Assuming `states` is a dictionary or can be indexed by `agent`
                value, _ = critic_policy(agent_states, None)  # Pass the states for the specific agent
                values.append(value)

            values = torch.stack(values)  # Combine the values for all agents
            values = values.squeeze()

            # Compute advantages (returns - values)
            advantages = returns - values.detach()

            # Actor update: Policy gradient loss
            action_probs = []
            log_probs = []
            actions = []

            for agent, actor_policy in self.actor_policies.items():
                action_mask = self.env.get_action_mask(agent)

                agent_obs = obs[agent]  # Assuming `obs` is indexed by agent
                agent_obs = ActorRNN.process_observation(agent_obs)
                action_prob, _ = actor_policy(agent_obs, None)  # Pass reshaped observation to RNN

                masked_action_prob = {}
                for key in action_prob:
                    # Mask the action probabilities
                    mask = torch.tensor(action_mask[key], dtype=torch.float32)
                    masked_action_prob[key] = action_prob[key] * mask
                    # Normalize probabilities after masking
                    masked_action_prob[key] = masked_action_prob[key] / masked_action_prob[key].sum()

                # Sample actions and compute log probabilities
                action_type_dist = Categorical(probs=masked_action_prob['action_type'])
                action_type = action_type_dist.sample()
                action_type_log_prob = action_type_dist.log_prob(action_type)
                
                if masked_action_prob['unit_id'].sum() > 0:
                    unit_id_dist = Categorical(probs=masked_action_prob['unit_id'])
                    unit_id = unit_id_dist.sample()
                    unit_id_log_prob = unit_id_dist.log_prob(unit_id)
                else:
                    unit_id = torch.tensor(0, dtype=torch.long)  # Ensure it's a tensor
                    unit_id_log_prob = torch.tensor(0.0)  # No meaningful log_prob; assign 0.0

                direction_dist = Categorical(probs=masked_action_prob['direction'])
                direction = direction_dist.sample()
                direction_log_prob = direction_dist.log_prob(direction)
                
                if masked_action_prob['city_id'].sum() > 0:
                    city_id_dist = Categorical(probs=masked_action_prob['city_id'])
                    city_id = city_id_dist.sample()
                    city_id_log_prob = city_id_dist.log_prob(city_id)
                else:
                    # Assign city_id as a tensor with the default value
                    city_id = torch.tensor(0, dtype=torch.long)  # Ensure it's a tensor
                    city_id_log_prob = torch.tensor(0.0)  # No meaningful log_prob; assign 0.0


                project_id_dist = Categorical(probs=masked_action_prob['project_id'])
                project_id = project_id_dist.sample()
                project_id_log_prob = project_id_dist.log_prob(project_id)

                # Store sampled actions
                actions.append({
                    'action_type': action_type,
                    'unit_id': unit_id,
                    'direction': direction,
                    'city_id': city_id,
                    'project_id': project_id
                })

                # Sum log probabilities for all action types
                total_log_prob = (action_type_log_prob +
                                unit_id_log_prob +
                                direction_log_prob +
                                city_id_log_prob +
                                project_id_log_prob)

                log_probs.append(total_log_prob)

            # Combine log probabilities into a single tensor
            log_probs = torch.stack(log_probs)

            actor_loss = -(log_probs * advantages).mean()  # Policy gradient loss

            # Zero out gradients for all actor optimizers
            for optimizer in actor_optimizers.values():
                optimizer.zero_grad()
            
            actor_loss.backward()

            for optimizer in actor_optimizers.values():
                optimizer.step()

            # Critic update: MSE loss
            critic_loss = F.mse_loss(values, returns)

            # Zero out gradients for all actor optimizers
            for optimizer in critic_optimizers.values():
                optimizer.zero_grad()
            
            critic_loss.backward()
            
            for optimizer in critic_optimizers.values():
                optimizer.step()

            
            if (iter + 1) % eval_interval == 0:

                self.env.reset()
                step=0
                while step < eval_steps:

                    for agent in self.env.agent_iter():
                        # sample action
                        # Get the current agent's observation
                        observation = self.env.observe(agent)
                        obs_tensor = ActorRNN.process_observation(observation)

                        # Get action probabilities from the actor policy
                        with torch.no_grad():
                            action_probs, _ = self.actor_policies[agent](obs_tensor, None)

                        # Sample action from probabilities
                        chosen_action = ProximalPolicyOptimization.sample_action(action_probs)
                        self.env.step(chosen_action)
                        self.env.render()
                        step+=1
                        if step >= eval_steps:
                            break

        return actor_loss.item(), critic_loss.item()

    def sample_trajectories(self,env, actor_policy, critic_policy, past_trajectory, n_agents, iter, agent):
        """
        Based off of Yu et al.'s 2022 paper on Recurrent MAPPO.
        Collect trajectories by interacting with the environment using recurrent actor and critic networks.

        Args:
            env: The environment to interact with.
            actor_policies: A dictionary mapping each agent to its actor policy function.
            critic_policies: A dictionary mapping each agent to its critic function.
            past_trajectory: The past trajectory for this agent

        Returns:
            List of trajectories. Each trajectory is a list of tuples containing:
                                                            
            current local observation                 ->  agent's policy ->  probability distribution over actions     ->    action
            actor hidden RNN from previous time step                         actor hidden RNN for current time step

            current state                             ->  value function  -> value estimate for agent
            critic hidden RNN from previous time step                        critic hidden RNN for current time step

            accumulate trajectory into:
            T = (state, obs, actor_hidden_state, critic_hidden_state, action, reward, next_state, next_observation).

            state vs observation: what is observed by all agents vs what is observed by a single agent
            actor hidden state vs critic hidden state: hidden state of an RNN, 
                but one takes into account present state and encoded history in deciding an action
                and the other takes into account present observation and encoded history in deciding a value estimate for 
        """

        #print("Running sample_trajectories")

        trajectories_next_step=[] #ends up being a list of num_agents length, where each element is what gets added to existing trajectory     
        
        #past_trajectory's last elements include state, obs, hidden actor, hidden critic, action, reward next time step, state next time step, obs next time step
        actor_hidden_state = past_trajectory[-6] 
        critic_hidden_state = past_trajectory[-5]
        state_t = past_trajectory[-2]
        obs_t = past_trajectory[-1]


        obs_t_tensor = ActorRNN.process_observation(obs_t)

        # Actor policy: get action distribution and next hidden state
        action_probs, next_actor_hidden = actor_policy(obs_t_tensor, actor_hidden_state)

        action_mask = self.env.get_action_mask(agent)

        # Apply action mask
        masked_action_probs = {}
        for key in action_probs:
            mask = torch.tensor(action_mask[key], dtype=torch.float32)
            masked_action_probs[key] = action_probs[key] * mask
            masked_action_probs[key] = masked_action_probs[key] / masked_action_probs[key].sum(dim=-1, keepdim=True)

        # Sample an action using masked probabilities
        action = ProximalPolicyOptimization.sample_action(masked_action_probs)

        # Step environment with all actions
        env.step(action)  # Pass a dictionary with the agent and its action
        reward_t = env.rewards[agent]
        sys.stdout.flush()
        self.reward_history[iter, agent] += reward_t

        done = env.dones[agent]
        next_obs = env.observe(agent)
        next_state = ProximalPolicyOptimization.get_global_state(env, n_agents)

        # Critic policy: get value and next critic hidden state
        value, next_critic_hidden = critic_policy(state_t, critic_hidden_state)

        trajectories_next_step=[
            state_t,
            obs_t, 
            next_actor_hidden,
            next_critic_hidden, 
            action,
            reward_t, 
            next_state,
            next_obs
        ]

        return trajectories_next_step

    def get_global_state(env, n_agents):

        obs_for_critic = []

        # Gather observations for all agents
        for agent_idx in range(n_agents):
            agent_obs = env.observe(env.agents[agent_idx])  # Get the local observation for the agent
            obs_for_critic.append(agent_obs)  

        # Critic policy: get value estimate and next hidden state
        critic_dict = {}
        # this is stupid naming, this is just the mask
        critic_visibility = env.get_full_masked_map()
        map_copy = env.map.copy()
        # do the masking
        critic_map = np.where(critic_visibility[:, :, np.newaxis].squeeze(2), map_copy, np.zeros_like(map_copy))
        # THIS MIGHT FUCK THINGS UP IN THE FUTURE. 
        critic_dict['map'] = critic_map
        critic_dict['units'] = None
        critic_dict['cities'] = None
        critic_dict['money'] = None
        for obs in obs_for_critic:
            for key in obs:
                if key != 'map':
                    if critic_dict[key] is None: 
                        critic_dict[key] = obs[key]
                    else:
                        critic_dict[key] = np.concatenate((critic_dict[key], obs[key]), axis=0) #idk if you can concatenate spaces.box
                else: 
                    pass

        for key in critic_dict: 
            critic_dict[key] = torch.tensor(critic_dict[key], dtype=torch.float32).flatten()
        state = torch.cat([critic_dict[key] for key in critic_dict]).unsqueeze(0)    
        return state

    def initialize_starting_trajectories(self,env, actor_policies, n_agents):
        """
        Initialize the starting trajectories for each agent.

        Args:
            env: The environment.
            actor_policies: A dictionary of actor policies for each agent.
            critic_policies: A dictionary of critic policies for each agent.

        Returns:
            A list of initial trajectories, one for each agent.
        """
        starting_trajectories = []

        n_agents = len(env.agents)
        initial_state = ProximalPolicyOptimization.get_global_state(env, n_agents) # Global state

        for agent_idx, agent in enumerate(env.agents):
            # Initial state and observation for each agent
            initial_observation = env.observe(agent)  # Local observation for the agent

            # Initialize hidden states for actor and critic networks
            input_size = actor_policies[agent_idx].hidden_size
            actor_hidden_state = torch.zeros(1, 1, input_size)  # Shape: (1, batch_size, hidden_size)
            critic_hidden_state = torch.zeros(1, 1, input_size)

            # Placeholder for the rest of the trajectory components
            initial_action = torch.zeros((5,), dtype=torch.float32)  # Adjust `action_space_size` as needed
            initial_reward = 0  # No reward has been received yet
            initial_next_state = initial_state  # Initial state is also the "next state"
            initial_next_observation = initial_observation  # Same for the observation

            # Create the initial trajectory structure
            starting_trajectory = [
                initial_state,             # State at time t
                initial_observation,       # Observation at time t
                actor_hidden_state,        # Actor hidden state
                critic_hidden_state,       # Critic hidden state
                initial_action,            # Action at time t (None at start)
                initial_reward,            # Reward at time t (0 at start)
                initial_next_state,        # Next state (same as initial state)
                initial_next_observation   # Next observation (same as initial observation)
            ]
            starting_trajectories.append(starting_trajectory)

        return starting_trajectories

    def sample_action(action_probs):

        action_type_dist = Categorical(probs=action_probs['action_type'])
        action_type = action_type_dist.sample().item()
        
        if action_probs['unit_id'].sum()>0:
            unit_id_dist = Categorical(probs=action_probs['unit_id'])
            unit_id = unit_id_dist.sample().item()
        else:
            unit_id = 0

        direction_dist = Categorical(probs=action_probs['direction'])
        direction = direction_dist.sample().item()

        if action_probs['city_id'].sum()>0:
            city_id_dist = Categorical(probs=action_probs['city_id'])
            city_id = city_id_dist.sample().item()
        else:
            city_id = 0

        project_id_dist = Categorical(probs=action_probs['project_id'])
        project_id = project_id_dist.sample().item()

        action = {
            'action_type': action_type,
            'unit_id': unit_id,
            'direction': direction,
            'city_id': city_id,
            'project_id': project_id,
        }
        return action
        

    def flatten_observation(self, observation):
        """
        Flattens a dictionary of observations into a single tensor.

        Args:
            observation (dict): A dictionary of observations.

        Returns:
            torch.Tensor: A flattened tensor of concatenated observation values.
        """
        tensors = []
        for key, value in observation.items():
            if isinstance(value, np.ndarray):  # Convert NumPy arrays to tensors
                value = torch.tensor(value, dtype=torch.float32)
            elif not isinstance(value, torch.Tensor):  # Skip non-tensor types
                continue
            tensors.append(value.flatten())  # Flatten each tensor
        return torch.cat(tensors)  # Concatenate into a single tensor

    def compute_returns(self, rewards, gamma=0.99):
            """
            Compute the discounted rewards-to-go (returns) for a given list of rewards.

            Args:
                rewards (torch.Tensor): A 1D tensor of rewards for a trajectory.
                gamma (float): Discount factor for future rewards. Default is 0.99.

            Returns:
                torch.Tensor: A 1D tensor of discounted rewards-to-go (returns).
            """
            returns = torch.zeros_like(rewards)
            discounted_sum = 0.0

            # Calculate the returns in reverse order
            for t in reversed(range(len(rewards))):
                discounted_sum = rewards[t] + gamma * discounted_sum
                returns[t] = discounted_sum

            return returns
    

class ActorRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_tiles):
        super(ActorRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc_action_type = nn.Linear(hidden_size, 7)
        self.fc_unit_id = nn.Linear(hidden_size, n_tiles)
        self.fc_direction = nn.Linear(hidden_size, 5)
        self.fc_city_id = nn.Linear(hidden_size, n_tiles)
        self.fc_project_id = nn.Linear(hidden_size, n_tiles)
            
    def process_observation(obs):
        if isinstance(obs, dict):
            processed_obs = []
            for key in obs:
                value = obs[key]
                if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                    tensor_value = torch.tensor(value, dtype=torch.float32).flatten()
                    processed_obs.append(tensor_value)
                else:
                    # Handle other types if necessary
                    pass
            obs_tensor = torch.cat(processed_obs).unsqueeze(0)
        elif isinstance(obs, np.ndarray) or isinstance(obs, torch.Tensor):
            # If obs is already a tensor or array, flatten and unsqueeze
            obs_tensor = torch.tensor(obs, dtype=torch.float32).flatten().unsqueeze(0)
        else:
            # Handle other types if necessary
            raise TypeError(f"Unsupported observation type: {type(obs)}")
        return obs_tensor


    def forward(self, observation, hidden_state):
        output, hidden_state = self.rnn(observation.unsqueeze(1), hidden_state)  # Add sequence dimension
        output = output.squeeze(1)  # Remove sequence dimension

        action_type_logits = self.fc_action_type(output)
        action_type_probs = F.softmax(action_type_logits, dim=-1)

        unit_id_logits = self.fc_unit_id(output)
        unit_id_probs = F.softmax(unit_id_logits, dim=-1)

        direction_logits = self.fc_direction(output)
        direction_probs = F.softmax(direction_logits, dim=-1)

        city_id_logits = self.fc_city_id(output)
        city_id_probs = F.softmax(city_id_logits, dim=-1)

        project_id_logits = self.fc_project_id(output)
        project_id_probs = F.softmax(project_id_logits, dim=-1)

        action_probs = {
            'action_type': action_type_probs,
            'unit_id': unit_id_probs,
            'direction': direction_probs,
            'city_id': city_id_probs,
            'project_id': project_id_probs,
        }

        return action_probs, hidden_state
    
class CriticRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Critic RNN that estimates the value of a state.

        Args:
            input_size: Size of the input observation.
            hidden_size: Size of the RNN hidden state.
        """
        super(CriticRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Output a single value
    
    def forward(self, observation, hidden_state):
        """
        Forward pass of the critic network.

        Args:
            observation: Input observation tensor of shape (batch_size, input_size).
            hidden_state: Hidden state of the RNN of shape (1, batch_size, hidden_size).

        Returns:
            value: Estimated value of shape (batch_size, 1).
            hidden_state: Updated hidden state of the RNN of shape (1, batch_size, hidden_size).
        """
        output, hidden_state = self.rnn(observation.unsqueeze(1), hidden_state)  # Add sequence dimension
        value = self.fc(output.squeeze(1))  # Remove sequence dimension
        return value, hidden_state