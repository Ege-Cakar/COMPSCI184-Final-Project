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
            actor_policies: Dict of agent_id -> actor network
            critic_policies: Dict of agent_id -> critic network
            lambdaa: Regularization coefficient.
            theta_inits: Initial policy parameters.
            n_iters: Number of training iterations.
            n_fit_trajectories: Number of trajectories for fitting the advantage function.
            n_sample_trajectories: Number of trajectories for optimizing the policy.
            max_steps: Number of steps per iteration.
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
            eval_interval: how often to evaluate
            eval_steps: how many steps to evaluate
        """
        actor_optimizers = {
            agent: torch.optim.Adam(policy.parameters(), lr=1e-3)
            for agent, policy in self.actor_policies.items()
        }

        critic_optimizers = {
            agent: torch.optim.Adam(policy.parameters(), lr=1e-3)
            for agent, policy in self.critic_policies.items()
        }

        theta_list = self.theta_inits

        for iter in range(self.n_iters):
            print("Training iteration:", iter)
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
                agent_states = states[agent]  # Assuming indexing by agent is possible
                value, _ = critic_policy(agent_states, None) 
                values.append(value)

            values = torch.stack(values) 
            values = values.squeeze()

            # Compute advantages (returns - values)
            advantages = returns - values.detach()

            # Actor update: Policy gradient loss
            action_probs = []
            log_probs = []
            acts = []

            for agent, actor_policy in self.actor_policies.items():
                action_mask = self.env.get_action_mask(agent)

                agent_obs = obs[agent]  
                agent_obs = ActorRNN.process_observation(agent_obs)
                action_prob, _ = actor_policy(agent_obs, None)  

                masked_action_prob = {}
                for key in action_prob:
                    mask = torch.tensor(action_mask[key], dtype=torch.float32)
                    masked_action_prob[key] = action_prob[key] * mask

                    # Handle zero-sum case: If no valid actions, choose NO_OP or fallback
                    mask_sum = masked_action_prob[key].sum()
                    if mask_sum.item() <= 0:
                        # Set all to zero
                        masked_action_prob[key] = torch.zeros_like(masked_action_prob[key])

                        # Determine NO_OP index for each key
                        if key == 'action_type':
                            no_op_index = 4  # NO_OP is 4
                        elif key == 'unit_id':
                            no_op_index = self.env.max_units_per_agent
                        elif key == 'city_id':
                            no_op_index = self.env.max_cities
                        elif key == 'project_id':
                            no_op_index = self.env.max_projects
                        else:
                            # direction fallback index
                            no_op_index = 4

                        # Assign 1.0 to NO_OP index
                        masked_action_prob[key][no_op_index] = 1.0
                        mask_sum = 1.0

                    masked_action_prob[key] = masked_action_prob[key] / mask_sum

                # Sample actions and compute log probabilities
                action_type_dist = Categorical(probs=masked_action_prob['action_type'])
                action_type = action_type_dist.sample()
                action_type_log_prob = action_type_dist.log_prob(action_type)
                
                if masked_action_prob['unit_id'].sum() > 0:
                    unit_id_dist = Categorical(probs=masked_action_prob['unit_id'])
                    unit_id = unit_id_dist.sample()
                    unit_id_log_prob = unit_id_dist.log_prob(unit_id)
                else:
                    unit_id = torch.tensor(0, dtype=torch.long)
                    unit_id_log_prob = torch.tensor(0.0)

                direction_dist = Categorical(probs=masked_action_prob['direction'])
                direction = direction_dist.sample()
                direction_log_prob = direction_dist.log_prob(direction)
                
                if masked_action_prob['city_id'].sum() > 0:
                    city_id_dist = Categorical(probs=masked_action_prob['city_id'])
                    city_id = city_id_dist.sample()
                    city_id_log_prob = city_id_dist.log_prob(city_id)
                else:
                    city_id = torch.tensor(0, dtype=torch.long)
                    city_id_log_prob = torch.tensor(0.0)

                project_id_dist = Categorical(probs=masked_action_prob['project_id'])
                project_id = project_id_dist.sample()
                project_id_log_prob = project_id_dist.log_prob(project_id)

                # Store sampled actions
                acts.append({
                    'action_type': action_type,
                    'unit_id': unit_id,
                    'direction': direction,
                    'city_id': city_id,
                    'project_id': project_id
                })

                # Sum log probabilities for all action components
                total_log_prob = (action_type_log_prob +
                                unit_id_log_prob +
                                direction_log_prob +
                                city_id_log_prob +
                                project_id_log_prob)

                log_probs.append(total_log_prob)

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
                        observation = self.env.observe(agent)
                        obs_tensor = ActorRNN.process_observation(observation)

                        with torch.no_grad():
                            action_probs, _ = self.actor_policies[agent](obs_tensor, None)

                        # Apply same logic to masked_action_probs
                        action_mask = self.env.get_action_mask(agent)
                        masked_action_probs = {}
                        for key in action_probs:
                            mask = torch.tensor(action_mask[key], dtype=torch.float32)
                            masked_action_probs[key] = action_probs[key] * mask

                            mask_sum = masked_action_probs[key].sum()
                            if mask_sum.item() <= 0:
                                # No valid actions, choose NO_OP
                                masked_action_probs[key] = torch.zeros_like(masked_action_probs[key])
                                if key == 'action_type':
                                    no_op_index = 4
                                elif key == 'unit_id':
                                    no_op_index = self.env.max_units_per_agent
                                elif key == 'city_id':
                                    no_op_index = self.env.max_cities
                                elif key == 'project_id':
                                    no_op_index = self.env.max_projects
                                else:
                                    no_op_index = 4
                                masked_action_probs[key][no_op_index] = 1.0
                                mask_sum = 1.0

                            masked_action_probs[key] = masked_action_probs[key] / mask_sum

                        chosen_action = ProximalPolicyOptimization.sample_action(masked_action_probs)
                        self.env.step(chosen_action)
                        self.env.render()
                        step += 1
                        if step >= eval_steps:
                            break

        return actor_loss.item(), critic_loss.item()

    def sample_trajectories(self, env, actor_policy, critic_policy, past_trajectory, n_agents, iter, agent):
        trajectories_next_step = []
        
        actor_hidden_state = past_trajectory[-6] 
        critic_hidden_state = past_trajectory[-5]
        state_t = past_trajectory[-2]
        obs_t = past_trajectory[-1]

        obs_t_tensor = ActorRNN.process_observation(obs_t)

        # Actor policy: get action distribution and next hidden state
        action_probs, next_actor_hidden = actor_policy(obs_t_tensor, actor_hidden_state)
        action_mask = env.get_action_mask(agent)

        masked_action_probs = {}
        for key in action_probs:
            mask = torch.tensor(action_mask[key], dtype=torch.float32)
            masked_action_probs[key] = action_probs[key] * mask

            mask_sum = masked_action_probs[key].sum()
            if mask_sum.item() <= 0:
                # No valid actions, choose NO_OP
                masked_action_probs[key] = torch.zeros_like(masked_action_probs[key])

                if key == 'action_type':
                    no_op_index = 4
                elif key == 'unit_id':
                    no_op_index = env.max_units_per_agent
                elif key == 'city_id':
                    no_op_index = env.max_cities
                elif key == 'project_id':
                    no_op_index = env.max_projects
                else:
                    no_op_index = 4
                masked_action_probs[key][no_op_index] = 1.0
                mask_sum = 1.0

            masked_action_probs[key] = masked_action_probs[key] / mask_sum

        # Sample an action using masked probabilities
        action = ProximalPolicyOptimization.sample_action(masked_action_probs)

        env.step(action)
        reward_t = env.rewards[agent]
        sys.stdout.flush()
        self.reward_history[iter, agent] += reward_t

        done = env.dones[agent]
        next_obs = env.observe(agent)
        next_state = ProximalPolicyOptimization.get_global_state(env, n_agents)

        # Critic policy
        value, next_critic_hidden = critic_policy(state_t, critic_hidden_state)

        trajectories_next_step = [
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
        for agent_idx in range(n_agents):
            agent_obs = env.observe(env.agents[agent_idx])
            obs_for_critic.append(agent_obs)  

        critic_dict = {}
        critic_visibility = env.get_full_masked_map()
        map_copy = env.map.copy()
        critic_map = np.where(critic_visibility[:, :, np.newaxis].squeeze(2), map_copy, np.zeros_like(map_copy))
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
                        critic_dict[key] = np.concatenate((critic_dict[key], obs[key]), axis=0)
                else:
                    pass

        for key in critic_dict: 
            critic_dict[key] = torch.tensor(critic_dict[key], dtype=torch.float32).flatten()
        state = torch.cat([critic_dict[key] for key in critic_dict]).unsqueeze(0)    
        return state

    def initialize_starting_trajectories(self, env, actor_policies, n_agents):
        starting_trajectories = []
        n_agents = len(env.agents)
        initial_state = ProximalPolicyOptimization.get_global_state(env, n_agents)

        for agent_idx, agent in enumerate(env.agents):
            initial_observation = env.observe(agent)
            input_size = actor_policies[agent_idx].hidden_size
            actor_hidden_state = torch.zeros(1, 1, input_size)
            critic_hidden_state = torch.zeros(1, 1, input_size)

            initial_action = torch.zeros((5,), dtype=torch.float32)
            initial_reward = 0
            initial_next_state = initial_state
            initial_next_observation = initial_observation

            starting_trajectory = [
                initial_state,           
                initial_observation,     
                actor_hidden_state,      
                critic_hidden_state,     
                initial_action,          
                initial_reward,          
                initial_next_state,      
                initial_next_observation 
            ]
            starting_trajectories.append(starting_trajectory)

        return starting_trajectories

    def sample_action(action_probs):
        action_type_dist = Categorical(probs=action_probs['action_type'])
        action_type = action_type_dist.sample().item()
        
        if action_probs['unit_id'].sum() > 0:
            unit_id_dist = Categorical(probs=action_probs['unit_id'])
            unit_id = unit_id_dist.sample().item()
        else:
            unit_id = 0

        direction_dist = Categorical(probs=action_probs['direction'])
        direction = direction_dist.sample().item()

        if action_probs['city_id'].sum() > 0:
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
        tensors = []
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                value = torch.tensor(value, dtype=torch.float32)
            elif not isinstance(value, torch.Tensor):
                continue
            tensors.append(value.flatten())
        return torch.cat(tensors)  

    def compute_returns(self, rewards, gamma=0.99):
        returns = torch.zeros_like(rewards)
        discounted_sum = 0.0
        for t in reversed(range(len(rewards))):
            discounted_sum = rewards[t] + gamma * discounted_sum
            returns[t] = discounted_sum
        return returns


class ActorRNN(nn.Module):
    def __init__(self, input_size, hidden_size, env):
        super(ActorRNN, self).__init__()
        self.hidden_size = hidden_size
        self.env = env
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc_action_type = nn.Linear(hidden_size, 7)
        # direction = 5
        self.fc_direction = nn.Linear(hidden_size, 5)
        # unit_id = max_units_per_agent + 1
        self.fc_unit_id = nn.Linear(hidden_size, self.env.max_units_per_agent + 1)
        # city_id = max_cities + 1
        self.fc_city_id = nn.Linear(hidden_size, self.env.max_cities + 1)
        # project_id = max_projects + 1
        self.fc_project_id = nn.Linear(hidden_size, self.env.max_projects + 1)
            
    def process_observation(obs):
        if isinstance(obs, dict):
            processed_obs = []
            for key in obs:
                value = obs[key]
                if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                    tensor_value = torch.tensor(value, dtype=torch.float32).flatten()
                    processed_obs.append(tensor_value)
            obs_tensor = torch.cat(processed_obs).unsqueeze(0)
        elif isinstance(obs, np.ndarray) or isinstance(obs, torch.Tensor):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).flatten().unsqueeze(0)
        else:
            raise TypeError(f"Unsupported observation type: {type(obs)}")
        return obs_tensor

    def forward(self, observation, hidden_state):
        output, hidden_state = self.rnn(observation.unsqueeze(1), hidden_state)
        output = output.squeeze(1)

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
        super(CriticRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, observation, hidden_state):
        output, hidden_state = self.rnn(observation.unsqueeze(1), hidden_state)
        value = self.fc(output.squeeze(1))
        return value, hidden_state