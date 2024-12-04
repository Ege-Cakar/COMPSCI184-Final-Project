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

    def train(self):
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

        #This code is taken from class
        theta_list = self.theta_inits

        starting_trajectories = self.initialize_starting_trajectories(self.env, self.actor_policies, self.critic_policies)

        for _ in range(self.n_iters):

            n_agents = len(self.actor_policies)

            trajectories = [[] for idx in range(n_agents)] # all trajectories, across all agents
            for _ in range(self.max_steps):

                for agent_idx in range(n_agents):
                    agent = self.env.agent_selection                
                    trajectories_next_step = self.sample_trajectories( #for a single agent
                        self.env,
                        self.actor_policies[agent_idx],
                        self.critic_policies[agent_idx],
                        trajectories[agent_idx]
                    )
                    trajectories[agent_idx].append(trajectories_next_step)

            A_hat_list = [self.fit(agent_fit_trajectories) for agent_fit_trajectories in trajectories]

            sample_trajectories = [[] for idx in range(n_agents)] # all trajectories, across all agents
            for _ in range(self.max_steps):

                for agent_idx in range(n_agents):
                    agent = self.env.agent_selection                
                    trajectories_next_step = self.sample_trajectories( #for a single agent
                        self.env,
                        self.actor_policies[agent_idx],
                        self.critic_policies[agent_idx],
                        trajectories[agent_idx]
                    )
                    sample_trajectories[agent_idx].append(trajectories_next_step)

            for agent_idx, theta in enumerate(theta_list):
                def objective(theta_opt):
                    total_objective = 0
                    for tau in sample_trajectories[agent_idx]:
                        for s, a, _r in tau:
                            pi_curr = self.pi(theta)(s, a)
                            pi_new = self.pi(theta_opt)(s, a)
                            total_objective += pi_new / pi_curr * A_hat_list[agent_idx](s, a) + self.lambdaa * torch.log(pi_new)
                    return total_objective / self.n_sample_trajectories
                
                theta_list[agent_idx] = self.optimize(self.actor_polices[agent_idx], self.critic_policies[agent_idx], sample_trajectories, learning_rate=1e-3)

        return theta_list
    
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
        #print(critic_map.shape)
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

    def initialize_starting_trajectories(env, actor_policies, critic_policies):
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

        n_agents = env.agents
        initial_state = ProximalPolicyOptimization.get_global_state(env, n_agents) # Global state

        for agent_idx, agent in enumerate(env.agents):
            # Initial state and observation for each agent
            initial_observation = env.observe(agent)  # Local observation for the agent

            # Initialize hidden states for actor and critic networks
            input_size = actor_policies[agent_idx].hidden_size
            actor_hidden_state = torch.zeros(1, 1, input_size)  # Shape: (1, batch_size, hidden_size)
            critic_hidden_state = torch.zeros(1, 1, input_size)

            # Placeholder for the rest of the trajectory components
            initial_action = None  # No action has been taken yet
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

    def sample_trajectories(env, actor_policy, critic_policy, past_trajectory):
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
        agent = env.agent_selection                

        trajectories_next_step=[] #ends up being a list of num_agents length, where each element is what gets added to existing trajectory     
        
        #past_trajectory's last elements include state, obs, hidden actor, hidden critic, action, reward next time step, state next time step, obs next time step
        actor_hidden_state = past_trajectory[-6] 
        critic_hidden_state = past_trajectory[-5]
        state_t = past_trajectory[-2]
        obs_t = past_trajectory[-1]

        # Actor policy: get action distribution and next hidden state
        action_probs, next_actor_hidden = actor_policy(obs_t, actor_hidden_state)
        action = ProximalPolicyOptimization.sample_action(action_probs)
        env.step(action)

        # Step environment with all actions
        agent = env.agent_selection
        env.step({agent: action})  # Pass a dictionary with the agent and its action
        reward_t = env.rewards[agent]

        done = env.dones[agent]
        next_obs = env.observe(agent)
        next_state = ProximalPolicyOptimization.get_global_state(env)

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

    def sample_action(action_probs):
        # Sample action components
        action_type_dist = Categorical(probs=action_probs['action_type'])
        action_type = action_type_dist.sample().item()

        unit_id_dist = Categorical(probs=action_probs['unit_id'])
        unit_id = unit_id_dist.sample().item()

        direction_dist = Categorical(probs=action_probs['direction'])
        direction = direction_dist.sample().item()

        city_id_dist = Categorical(probs=action_probs['city_id'])
        city_id = city_id_dist.sample().item()

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


    def fit(trajectories):
        """
        Fit the advantage function from the given trajectories.

        Args:
            trajectories: A list of trajectories. Each trajectory is a list of (state, action, reward) tuples.

        Returns:
            A_hat: A callable advantage function A_hat(s, a).
        """
        def compute_returns(rewards, gamma=0.99):
            """
            Compute the discounted returns for a trajectory.

            Args:
                rewards: A list of rewards for a single trajectory.
                gamma: Discount factor for future rewards.

            Returns:
                Discounted returns.
            """
            returns = []
            discounted_sum = 0
            for r in reversed(rewards):
                discounted_sum = r + gamma * discounted_sum
                returns.insert(0, discounted_sum)
            return returns

        states, actions, rewards = [], [], []
        for trajectory in trajectories:
            for s, a, r in trajectory:
                states.append(s)
                actions.append(a)
                rewards.append(r)

        # Compute returns for each trajectory
        all_returns = []
        for trajectory in trajectories:
            rewards = [r for _, _, r in trajectory]
            all_returns.extend(compute_returns(rewards))

        states = np.array(states)
        actions = np.array(actions)
        returns = np.array(all_returns)

        # Estimate the value function V(s) as the average return for each state
        unique_states = np.unique(states, axis=0)
        state_to_value = {tuple(s): returns[states == s].mean() for s in unique_states}
        V = lambda s: state_to_value.get(tuple(s), 0)

        # Define the advantage function A(s, a) = Q(s, a) - V(s)
        def A_hat(s, a):
            return returns[(states == s) & (actions == a)].mean() - V(s)

        return A_hat
    
    def optimize(
        actor_policy,
        critic_policy,
        trajectories,
        learning_rate,
        gamma=0.99,
        lam=0.95,
        batch_size=64,
        chunk_length=10,
        num_epochs=4
    ):
        """
        Implements the optimization step for Recurrent-MAPPO.

        Args:
            actor_policy: The actor RNN policy network.
            critic_policy: The critic RNN network.
            trajectories: Collected trajectories from the environment.
            learning_rate: Learning rate for Adam optimizer.
            gamma: Discount factor for future rewards.
            lam: GAE lambda for advantage computation.
            batch_size: Batch size for optimization.
            chunk_length: Length of trajectory chunks.
            num_epochs: Number of optimization epochs.

        Returns:
            Updated actor and critic policies.
        """
        # Initialize optimizers for actor and critic
        actor_optimizer = torch.optim.Adam(actor_policy.parameters(), lr=learning_rate)
        critic_optimizer = torch.optim.Adam(critic_policy.parameters(), lr=learning_rate)

        # Extract data from trajectories
        states = torch.stack([t[0] for t in trajectories])
        observations = torch.stack([t[1] for t in trajectories])
        actions = torch.stack([t[4] for t in trajectories])
        rewards = torch.tensor([t[5] for t in trajectories])
        next_states = torch.stack([t[6] for t in trajectories])
        dones = torch.tensor([t[7] for t in trajectories], dtype=torch.float32)

        # Compute value estimates
        with torch.no_grad():
            values, _ = critic_policy(states, None)  # Critic values
            next_values, _ = critic_policy(next_states, None)  # Next state values
            values = values.squeeze()
            next_values = next_values.squeeze()

            # Compute TD(λ) targets and advantages using GAE
            advantages = []
            gae = 0
            td_targets = []
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + (1 - dones[t]) * gamma * next_values[t] - values[t]
                gae = delta + gamma * lam * (1 - dones[t]) * gae
                advantages.insert(0, gae)
                td_targets.insert(0, rewards[t] + gamma * (1 - dones[t]) * next_values[t])

            advantages = torch.tensor(advantages, dtype=torch.float32)
            td_targets = torch.tensor(td_targets, dtype=torch.float32)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Split trajectory into chunks of length L
        num_chunks = len(trajectories) // chunk_length
        chunks = []
        for i in range(0, len(trajectories), chunk_length):
            chunks.append(
                (
                    states[i : i + chunk_length],
                    observations[i : i + chunk_length],
                    actions[i : i + chunk_length],
                    advantages[i : i + chunk_length],
                    td_targets[i : i + chunk_length],
                )
            )

        # Optimization loop
        for _ in range(num_epochs):
            for _ in range(batch_size):
                # Randomly sample mini-batch of chunks
                minibatch = np.random.sample(chunks, batch_size)

                for (state_chunk, obs_chunk, action_chunk, adv_chunk, td_chunk) in minibatch:
                    # Reset hidden states for actor and critic
                    actor_hidden_state = torch.zeros(1, 1, actor_policy.hidden_size)
                    critic_hidden_state = torch.zeros(1, 1, critic_policy.hidden_size)

                    # Compute action probabilities and values
                    action_probs, actor_hidden_state = actor_policy(obs_chunk, actor_hidden_state)
                    dist = torch.distributions.Categorical(action_probs)
                    log_probs = dist.log_prob(action_chunk)

                    values, critic_hidden_state = critic_policy(state_chunk, critic_hidden_state)
                    values = values.squeeze()

                    # Compute actor loss (PPO surrogate loss with clipping)
                    old_log_probs = log_probs.detach()
                    ratio = torch.exp(log_probs - old_log_probs)
                    surrogate1 = ratio * adv_chunk
                    surrogate2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv_chunk
                    actor_loss = -torch.min(surrogate1, surrogate2).mean()

                    # Compute critic loss (MSE between values and TD targets)
                    critic_loss = F.mse_loss(values, td_chunk)

                    # Backpropagation and optimization
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    actor_loss.backward()
                    critic_loss.backward()
                    actor_optimizer.step()
                    critic_optimizer.step()

        return actor_policy, critic_policy

    def compute_loss(actor_policy, trajectories, old_log_probs, advantages, clip_epsilon=0.2):
        """
        Compute the PPO surrogate loss.

        Args:
            actor_policy: The actor neural network.
            trajectories: Collected trajectories.
            old_log_probs: Log probabilities of actions taken, from the behavior policy.
            advantages: Estimated advantages.
            clip_epsilon: Clipping parameter for PPO.

        Returns:
            The computed loss.
        """
        # Compute new log probabilities
        new_log_probs = []
        for (state, action, _, _) in trajectories:
            action_probs, _ = actor_policy(state, hidden_state=None)
            dist = Categorical(action_probs)
            new_log_prob = dist.log_prob(action)
            new_log_probs.append(new_log_prob)

        new_log_probs = torch.stack(new_log_probs)
        old_log_probs = torch.stack(old_log_probs)
        advantages = torch.stack(advantages)

        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped surrogate objective
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        loss = -torch.min(surrogate1, surrogate2).mean()

        return loss

    

class ActorRNN(nn.Module):
    def __init__(self, input_size, hidden_size, max_units_per_agent, max_cities, max_projects):
        super(ActorRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc_action_type = nn.Linear(hidden_size, 7)
        self.fc_unit_id = nn.Linear(hidden_size, max_units_per_agent)
        self.fc_direction = nn.Linear(hidden_size, 4)
        self.fc_city_id = nn.Linear(hidden_size, max_cities)
        self.fc_project_id = nn.Linear(hidden_size, max_projects)
    
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
