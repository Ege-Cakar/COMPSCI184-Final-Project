import wandb
import matplotlib.pyplot as plt
import numpy as np
import sys
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
import random
from rnn import ActorRNN, CriticRNN
import wandb


from civ import Civilization

# Type aliases for readability
State = TypeVar("State")  # Represents the state type
Action = TypeVar("Action")  # Represents the action type

class ProximalPolicyOptimization:
    def __init__(self, env, actor_policies, critic_policies, lambdaa, step_max, T, batch_size, K, device):
        """
        Initialize PPO with environment and hyperparameters.

        Args:
            env: The environment for training.
            pi: Policy function that maps parameters to a function defining action probabilities.
            lambdaa: Regularization coefficient.
            theta_init: Initial policy parameters.
            step_max: Number of training iterations.
            T: Number of iterations to run trajectory for
        """
        self.device = device
        self.env = env
        self.actor_policies = {
            agent: policy.to(self.device) for agent, policy in actor_policies.items()
        }
        self.critic_policies = critic_policies.to(self.device)
        self.lambdaa = lambdaa
        self.step_max = step_max
        self.T = T
        self.batch_size = batch_size
        self.K = K


    def train(self):
        """
        Proximal Policy Optimization (PPO) implementation.

        Args:
            env: The environment in which the agent will act.
            actor_policies: RNN defined in test.py
            critic_policies: RNN defined in test.py
            λ: Regularization coefficient to penalize large changes in the policy.
            theta_inits: Initial parameters of the policy, conists of k1 through k10 and epsilon (environmental impact)
            step_max: Number of training iterations.
        
        Returns:
            Trained policy parameters, theta.
        """
        
        #each agent in the environment is assigned a separate optimizer for both their actor policy and critic policy networks.
        actor_optimizers = {
            agent: torch.optim.Adam(policy.parameters(), lr=1e-3)
            for agent, policy in self.actor_policies.items()
        }

        critic_optimizers = torch.optim.Adam(self.critic_policies.parameters(), lr=1e-3)

        
        keys = [
            "P_progress",
            "P_completion",
            "C_tiles",
            "C_cities",
            "L_cities",
            "C_units",
            "L_units",
            "delta_GDP",
            "delta_Energy",
            "C_resources",
            "E_impact",
            "Stalling",
            "Entropy"
            ]
        
        eval_interval = self.step_max
        eval_steps = self.T
        config = wandb.config

        cumulative_rewards = torch.zeros((len(self.env.agents), self.step_max), device=self.device) # List to store cumulative rewards per iteration
        reward_components = {agent: {key: [0] * self.step_max for key in keys} for agent in self.env.agents}

        for step in range(self.step_max):
            print(f"Training iteration: {step}")
            sys.stdout.flush()

            D = []

            for i in range(self.batch_size):

                print(f"On batch {i}")

                self.env.reset()   
                
                trajectories, cumulative_rewards, reward_components = self.generate_all_trajectories(cumulative_rewards,reward_components, step)
                
                old_action_probs = self.compute_old_action_probs(trajectories)
                old_action_probs = [{k: v.detach() for k,v in a.items()} for a in old_action_probs]

                # Compute advantages
                A_hat = self.fit(trajectories)

                D.append(trajectories)

            for minibatch in range(int(self.K)): #for now, just a single trajectory

                random_mini_batch = random.choice(D) #TO CHANGE: mini batch is 1 trajectory rn
                chunk_size = 9*5 # TO CHANGE: just doing every 5 time steps

                data_chunks = [
                    [row[i:i+chunk_size] for row in random_mini_batch]
                    for i in range(0, len(random_mini_batch[0]), chunk_size)
                ]
                # Take the first data chunk
                data_chunk = data_chunks[0]

                # If your step structure is [state, obs, actor_hidden, critic_hidden, action, reward, next_state, next_obs, value]
                actor_hiddens = [agent_chunk[2] for agent_chunk in data_chunk]
                critic_hiddens = data_chunk[0][3].to(self.device)

                critic_hiddens = critic_hiddens.squeeze(0)    


                for data_chunk in data_chunks:
                    # update RNN hidden states for π and V from first hidden state in data chunk and propagate
                    probs, actor_hiddens, values,critic_hiddens= self.updateRNN(data_chunk, actor_hiddens, critic_hiddens)

            # ACTOR LOSS UPDATES
            self.actor_adam(trajectories, actor_optimizers, old_action_probs, probs, A_hat)
            old_action_probs = probs
            self.critic_adam(trajectories, critic_optimizers, values)

            self.evaluate(step, eval_interval, eval_steps)
        
        fig, axes = plt.subplots(len(self.env.agents), 1, figsize=(8, 4 * len(self.env.agents)))

        for agent in self.env.agents:
            ax = axes[agent]
            for reward_type in keys:
                ax.plot(range(self.step_max),reward_components[agent][reward_type], label = reward_type)
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Cumulative Reward")
                ax.set_title(f"Agent{agent}")
                ax.grid()   
                ax.legend()
        plt.tight_layout
        if self.step_max >=10:
            plt.savefig("outputs/cumulative_reward_component.png")  # Save as PNG
        plt.show()

        plt.figure(figsize=(10, 6))
        for agent in range(len(self.env.agents)):
            plt.plot(range(self.step_max), cumulative_rewards[agent, :], label=f"Agent {agent }")
        plt.xlabel("Training Iterations")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Reward Over Training Iterations")
        plt.legend()
        plt.grid()
        if self.step_max >=10:
            plt.savefig("outputs/cumulative_reward.png")  # Save as PNG
        plt.show()

        # Data to save
        if self.step_max >=10:
            data = f"Hyperparameters:\n Ran for {self.step_max} iterations \n Each iteration runs {self.batch_size} trajectories \n Each trajectory contains {self.T} time steps"

            # Save to a text file
            with open("outputs/hyperparams.txt", "w") as file:
                file.write(data)
            return None
    
    def generate_all_trajectories(self, cumulative_rewards,reward_components, step):
        trajectories = self.initialize_starting_trajectories(self.env, self.actor_policies)
                
        t_ag=0
        for agent in self.env.agent_iter():
            trajectories_next_step, agent_reward_components = self.generate_single_trajectory( #for a single agent
                self.env,
                self.actor_policies[agent],
                self.critic_policies,
                trajectories[agent],
                agent
            )
            cumulative_rewards[agent, step]+=trajectories_next_step[-4] #for plotting

            for key, value in agent_reward_components.items():
                reward_components[agent][key][step]+=value

            trajectories[agent].extend(trajectories_next_step)
            t_ag+=1

            if t_ag >= self.T*len(self.env.agents):
                break
            #shape: # agents x length of trajectory
        
        return trajectories,cumulative_rewards, reward_components
    
    def generate_single_trajectory(self,env,actor_policy,critic_policy,past_trajectory,agent):
            
        trajectories_next_step=[] #ends up being a list of num_agents length, where each element is what gets added to existing trajectory     
        
        #past_trajectory's last elements include state, obs, hidden actor, hidden critic, action, reward next time step, state next time step, obs next time step
        actor_hidden_state = past_trajectory[-7] 
        critic_hidden_state = past_trajectory[-6]
        state_t = past_trajectory[-3]
        obs_t_dict = past_trajectory[-2]
        obs_t = self.flatten_observation(obs_t_dict).to(self.device)  # convert dict to tensor
        obs_t_input = obs_t.unsqueeze(0).unsqueeze(0) # This line is here twice? 

        # Prepare inputs for RNNs: (batch_size=1, seq_len=1, input_size)
        obs_t_input = obs_t.unsqueeze(0).unsqueeze(0)  # (1,1,input_size)
        state_t_input = state_t.unsqueeze(0).unsqueeze(0)  # (1,1,input_size_critic)


        # Actor policy: get action distribution and next hidden state
        action_probs, next_actor_hidden = actor_policy(obs_t_input, actor_hidden_state)
        action = self.sample_action(action_probs)

        # Step environment with all actions
        agent = env.agent_selection
        env.step(action)  # Pass a dictionary with the agent and its action
        reward_t , reward_components= env.rewards[agent]

        done = env.dones[agent]
        next_obs = env.observe(agent)
        next_state = self.get_global_state(env)

        # Critic policy: get value and next critic hidden state
        value, next_critic_hidden = critic_policy(state_t_input, critic_hidden_state)

        trajectories_next_step=[
            state_t,
            obs_t, 
            next_actor_hidden,
            next_critic_hidden, 
            action,
            reward_t, 
            next_state,
            next_obs,
            value
        ]

        return trajectories_next_step, reward_components
    
    def updateRNN(self,trajectories, initial_actor_hidden, initial_critic_hidden):
        first_agent_trajectory = trajectories[0]
        global_states = [
            t for i, t in enumerate(first_agent_trajectory)
            if (i % 9) == 0
        ]
        global_states = torch.stack(global_states, dim=0)  # shape: [T, state_dim]

        obs = torch.stack([
            self.flatten_observation(t)
            for agent_trajectories in trajectories
            for i, t in enumerate(agent_trajectories)
            if (i % 9) == 1  # Keep only the observation at index 1 of each 9-element chunk
        ])
        
        across_agent_probs, across_agent_final_actor_hiddens=[],[]
        global_states.unsqueeze(0)

        
        values, final_critic_hiddens = self.critic_policies(global_states, initial_critic_hidden)
        
        #update RNN hidden states for π and V from first hidden state in data chunk
        for agent in self.env.agents:

            agent_obs = obs[agent]

            actor_input = agent_obs.unsqueeze(0).unsqueeze(0)

            probs, final_actor_hidden = self.actor_policies[agent](actor_input, initial_actor_hidden[agent])
            across_agent_probs.append(probs)
            across_agent_final_actor_hiddens.append(final_actor_hidden)


        return across_agent_probs, across_agent_final_actor_hiddens, values, final_critic_hiddens
    
    def fit(self, trajectories,discount = 1.0, lam = 0.95):
        # Process trajectories
        rewards = [[t for i, t in enumerate(agent_trajectories) if (i % 9) == 5]
                   for agent_trajectories in trajectories]

        # Convert to a 2D tensor
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        first_agent_trajectory = trajectories[0]
        global_states = [
            t for i, t in enumerate(first_agent_trajectory)
            if (i % 9) == 0
        ]
        global_states = torch.stack(global_states, dim=0).to(self.device)  # shape: [T, state_dim]

        # Compute discounted returns-to-go
        returns = self.compute_returns(rewards, discount)  # Shape: (n_agents * T,)
        critic_hiddens = [
            t.to(self.device) for i, t in enumerate(first_agent_trajectory)
            if (i % 9) == 3
        ]
        critic_hiddens = torch.stack(critic_hiddens, dim=0).to(self.device)  # shape: [T, state_dim]
        critic_hiddens = critic_hiddens.squeeze(1).squeeze(1).unsqueeze(0)


        # Run the centralized critic to get values for each timestep.
        # Suppose your critic expects input shape: [T, state_dim], and hidden: [1, T, hidden_dim] or similar.
        # Adjust as necessary based on your CriticRNN definition.
        values, _ = self.critic_policies(global_states.unsqueeze(1), critic_hiddens)
        # Suppose values shape: [T, 1] after forward pass


        values = values.squeeze(-1)  # Now values: [T]


        # Compute advantages using Generalized Advantage Estimation (GAE)
        advantages = torch.zeros_like(rewards)  # [n_agents, T]
        gae = 0.0
        for t in reversed(range(self.T)):
            next_value = values[t+1,0].item() if t < self.T - 1 else 0.0
            # Use mean of rewards across agents or handle differently
            # If you prefer summation or another aggregation, do so here
            mean_reward = rewards[:, t].mean().item()
            delta = mean_reward + discount * next_value - values[t,0].item()
            gae = delta + discount * lam * gae
            # Assign the same advantage to all agents
            advantages[:, t] = gae

        # Normalize advantages
        flat_adv = advantages.flatten()
        advantages = (advantages - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        return advantages
    
    def compute_log_prob(self, distributions, action):
        # 'distributions' might be a dict or tuple containing separate distributions for each action dimension.
        # 'action' could be a dict or tuple with the chosen actions for each dimension.

        # Example assuming 'distributions' is a dict of Torch distributions and
        # 'action' is a dictionary with keys matching the distributions.
        log_prob = 0.0
        log_prob += distributions['action_type'].log_prob(torch.tensor(action['action_type']))
        log_prob += distributions['unit_id'].log_prob(torch.tensor(action['unit_id']))
        log_prob += distributions['direction'].log_prob(torch.tensor(action['direction']))
        log_prob += distributions['city_id'].log_prob(torch.tensor(action['city_id']))
        log_prob += distributions['project_id'].log_prob(torch.tensor(action['project_id']))
        return log_prob

    def actor_adam(self, trajectories, actor_optimizers, old_action_probs, new_action_probs, A_hat):
        obs = torch.stack([
            self.flatten_observation(t)
            for agent_trajectories in trajectories
            for i, t in enumerate(agent_trajectories)
            if (i % 9) == 1  # Keep only the observation at index 1 of each 9-element chunk
        ])

   
        actions = []
        for agent_trajectories in trajectories:
            # Extract actions (every 9th element starting at index 4)
            agent_actions = [t for i, t in enumerate(agent_trajectories) if i % 9 == 4]
            actions.append(agent_actions)
        

        for agent in self.env.agents:
            # Extract the per-agent trajectories, advantages, obs, actions
            agent_trajectory = trajectories[agent]
            agent_advantages = A_hat[agent]
            agent_obs = obs[agent]
            agent_actions = actions[agent]  # actions should be a tensor or dict per timestep

            new_log_probs = []
            old_log_probs = []

            # For every time step
            for t in range(self.T):
                # Retrieve the agent's chosen action at time t
                # If you've stored them as dictionaries:
                chosen_action = {
                    'action_type': agent_actions[t]['action_type'],
                    'unit_id': agent_actions[t]['unit_id'],
                    'direction': agent_actions[t]['direction'],
                    'city_id': agent_actions[t]['city_id'],
                    'project_id': agent_actions[t]['project_id']
                }
                # Compute the log probability from the new and old policies
                new_lp = self.compute_log_prob(new_action_probs[agent], chosen_action)
                old_lp = self.compute_log_prob(old_action_probs[agent], chosen_action)

                new_log_probs.append(new_lp)
                old_log_probs.append(old_lp)

            # Convert lists to tensors
            new_log_probs = torch.stack(new_log_probs)
            old_log_probs = torch.stack(old_log_probs)

            # Compute ratio and PPO loss as usual
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
            actor_loss = -torch.min(ratio * agent_advantages, clipped_ratio * agent_advantages).mean()

            actor_optimizers[agent].zero_grad()
            actor_loss.backward()
            actor_optimizers[agent].step()

    def critic_adam(self, trajectories, critic_optimizers, across_agent_values, discount=1.0, epsilon=0.1):
        """
        Perform critic updates using a single shared critic and PPO clipped value loss.

        Args:
            trajectories: List of trajectories for all agents.
            across_agent_values: Predicted values for all agents, presumably stacked or listed per agent.
            gamma: Discount factor.
            epsilon: Clipping parameter.
        """
        # Extract states, rewards, and compute returns
        states = torch.stack([
            t.to(self.device)
            for agent_trajectories in trajectories
            for i, t in enumerate(agent_trajectories)
            if (i % 9) == 0  # Keep only the 3rd element (index 2 of each 9-element chunk)
        ])
        rewards = torch.tensor([[t for i, t in enumerate(agent_trajectories) if (i % 9) == 5]
                   for agent_trajectories in trajectories], dtype=torch.float32)


        returns = self.compute_returns(rewards, discount)  # Shape: [n_agents, T]

        # across_agent_values should be a list of value tensors, one per agent, each shape [T, 1].
        # Let's stack them into a single tensor:
        # across_agent_values is currently a list indexed by agent, each an RNN output for that agent.
        # Convert to a tensor [n_agents, T, 1]

        # Flatten both returns and values
        flat_values = across_agent_values.view(-1).to(self.device)
        flat_returns = returns.view(-1).to(self.device)

        # Compute unclipped and clipped value losses
        value_loss_unclipped = (flat_values - flat_returns) ** 2
        clipped_values = torch.clamp(flat_values, min=(flat_returns - epsilon), max=(flat_returns + epsilon))
        value_loss_clipped = (clipped_values - flat_returns) ** 2

        # PPO value loss
        critic_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

        # Single backward and optimizer step
        critic_optimizers.zero_grad()
        critic_loss.backward()
        critic_optimizers.step()
    






    # def critic_adam(self, trajectories, critic_optimizers, across_agent_values, gamma=0.99, epsilon=0.1):
    #     """
    #     Perform critic updates using the PPO clipped value loss.

    #     Args:
    #         trajectories: List of trajectories for all agents.
    #         critic_optimizers: Dictionary of optimizers for each agent's critic network.
    #         critic_hidden: Dictionary of critic hidden states for each agent.
    #         gamma: Discount factor for returns.
    #         epsilon: Clipping parameter for value loss.
    #     """
    #     # Extract states, rewards, and compute returns
    #     states = torch.stack([
    #         t
    #         for agent_trajectories in trajectories
    #         for i, t in enumerate(agent_trajectories)
    #         if (i % 9) == 0  # Keep only the 3rd element (index 2 of each 9-element chunk)
    #     ])
    #     rewards = torch.tensor([[t for i, t in enumerate(agent_trajectories) if (i % 9) == 5]
    #                for agent_trajectories in trajectories], dtype=torch.float32)
    #     returns = self.compute_returns(rewards, gamma)  # Discounted returns-to-go

    #     for agent in self.env.agents:
    #         # Extract agent-specific data
    #         agent_returns = returns[agent]  # Returns for the agent

    #         values = across_agent_values[agent]

    #         # Compute unclipped and clipped value losses
    #         value_loss_unclipped = (values.squeeze(1) - agent_returns) ** 2
    #         clipped_values = torch.clamp(
    #             values.squeeze(1),
    #             min=(agent_returns - epsilon),
    #             max=(agent_returns + epsilon),
    #         )
    #         value_loss_clipped = (clipped_values - agent_returns) ** 2

    #         # Compute critic loss by taking the maximum loss
    #         agent_critic_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()

    #         # Backpropagation and optimizer step
    #         critic_optimizers[agent].zero_grad()
    #         agent_critic_loss.backward()
    #         critic_optimizers[agent].step()



    def evaluate(self, step, eval_interval, eval_steps):
        """
        Evaluate the trained policies on the environment.

        Args:
            step (int): Current training step.
            eval_interval (int): Interval at which evaluation is performed.
            eval_steps (int): Number of steps to evaluate.
        """
        if (step + 1) % eval_interval == 0:
            print("Starting evaluation...")
            self.env.reset()
            step_counter = 0

            # Initialize hidden states for all agents
            actor_hidden_states = {
                agent: torch.zeros(1, 1, self.actor_policies[agent].hidden_size) 
                for agent in self.env.agents
            }

            action_types=[[]for agent in self.env.agents]
                    # Action constants: 
            # self.MOVE_UNIT = 0
            # self.ATTACK_UNIT = 1
            # self.FOUND_CITY = 2
            # self.ASSIGN_PROJECT = 3
            # self.NO_OP = 4
            # self.BUY_WARRIOR = 5
            # self.BUY_SETTLER = 6

            # Loop over agent interactions in the environment
            for agent in self.env.agent_iter():
                print(f"Step {step_counter}, Agent {agent}")
                sys.stdout.flush()

                # Observe and process the observation
                observation = self.env.observe(agent)
                obs_tensor = ActorRNN.process_observation(observation).unsqueeze(0).unsqueeze(0)

                with torch.no_grad():
                    # Use the trained policy to compute action probabilities
                    action_probs, actor_hidden_states[agent] = self.actor_policies[agent](
                        obs_tensor, actor_hidden_states[agent]
                    )
                    # Sample an action
                    chosen_action = self.sample_action(action_probs)

                action_types[agent].append(chosen_action['action_type'])

                # Step the environment with the chosen action
                self.env.step(chosen_action)
                self.env.render()

                # Increment step counter when all agents have acted
                if agent == self.env.agents[-1]:
                    step_counter += 1

                # Break evaluation after reaching the desired number of steps
                if step_counter >= eval_steps:
                    break

            action_labels = [
                "MOVE_UNIT", "ATTACK_UNIT", "FOUND_CITY", 
                "ASSIGN_PROJECT", "NO_OP", "BUY_WARRIOR", "BUY_SETTLER"
            ]
            fig, axes = plt.subplots(len(self.env.agents), 1, figsize=(8, 4 * len(self.env.agents)))

            for agent in self.env.agents:
                ax = axes[agent]

                for action in range(7):
                    action_occurences = np.where(np.array(action_types[agent])==action)
                    sys.stdout.flush()
                    hist, bin_edges = np.histogram(action_occurences, bins=5)
                    # Convert histogram to line plot data
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    ax.plot(bin_centers, hist, marker='o', linestyle='-', label=action_labels[action])
                
                ax.set_title(f"Agent {agent}")
                ax.set_xlabel("Evaluation Steps")
                ax.set_ylabel("Action Counts")
                ax.legend()
            plt.tight_layout()
            if self.step_max >=10:
                plt.savefig("outputs/actions_eval.png")
            plt.show()
            print("Evaluation complete.")

    def compute_old_action_probs(self, trajectories):
        """
        Creates a copy of the actor policies for all agents, with the same structure and parameters.

        Returns:
            dict: A dictionary containing the copied actor policies for each agent.
        """
        obs = torch.stack([
            self.flatten_observation(t).to(self.device)
            for agent_trajectories in trajectories
            for i, t in enumerate(agent_trajectories)
            if (i % 9) == 1  # Keep only the observation at index 1 of each 9-element chunk
        ])
        actor_hidden_states = torch.stack([
            t.to(self.device) if isinstance(t, torch.Tensor) else torch.tensor(t, device=self.device)
            for agent_trajectories in trajectories
            for i, t in enumerate(agent_trajectories)
            if (i % 9) == 2  # Keep only the actor hidden states
        ])
                    

        old_actor_policies = {
            agent: type(self.actor_policies[agent])(
                self.actor_policies[agent].rnn.input_size,
                self.actor_policies[agent].hidden_size,
                self.actor_policies[agent].fc_unit_id.out_features,
                self.actor_policies[agent].fc_city_id.out_features,
                self.actor_policies[agent].fc_project_id.out_features,
                self.actor_policies[agent].device,
            ) for agent in self.env.agents
        }

        # Copy the parameters
        for agent in self.env.agents:
            old_actor_policies[agent].load_state_dict(self.actor_policies[agent].state_dict())
            
        across_agents_actions_probs =[]
        for agent in self.env.agents:   
            obs_agent = obs[agent].unsqueeze(0).unsqueeze(0)  

            action_probs, _ = old_actor_policies[agent](obs_agent, actor_hidden_states[agent])  # Actor hidden state
            across_agents_actions_probs.append(action_probs)

        return across_agents_actions_probs

    def compute_log_prob(self, action_probs, action):
        total_log_prob = 0.0
        for key, probs in action_probs.items():
            dist = Categorical(probs=probs)
            selected_action = torch.tensor(action[key], dtype=torch.long)
            selected_action = selected_action.to(self.device)
            log_prob = dist.log_prob(selected_action)
            total_log_prob += log_prob
        return total_log_prob
        
    def get_global_state(self, env):
        # Gather all agent observations to construct a global state
        obs_for_critic = []
        for agent in self.env.agents:
            agent_obs = self.env.observe(agent)
            obs_for_critic.append(agent_obs)

        critic_dict = {}
        critic_visibility = env.get_full_masked_map()
        map_copy = env.map.copy()
        critic_map = np.where(critic_visibility[:, :, np.newaxis].squeeze(2), map_copy, np.zeros_like(map_copy))
        critic_dict['map'] = torch.tensor(critic_map, dtype=torch.float32).flatten()

        # Concatenate units, cities, money across agents
        units_list = []
        cities_list = []
        money_list = []
        for obs in obs_for_critic:
            units_list.append(torch.tensor(obs['units'], dtype=torch.float32).flatten())
            cities_list.append(torch.tensor(obs['cities'], dtype=torch.float32).flatten())
            money_list.append(torch.tensor(obs['money'], dtype=torch.float32).flatten())

        critic_dict['units'] = torch.cat(units_list)
        critic_dict['cities'] = torch.cat(cities_list)
        critic_dict['money'] = torch.cat(money_list)

        state = torch.cat([critic_dict[k] for k in critic_dict]).float().to(self.device)
        return state

    def initialize_starting_trajectories(self,env, actor_policies):
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
        initial_state = self.get_global_state(env) # Global state

        for agent in self.env.agents:
            # Initial state and observation for each agent
            initial_observation = env.observe(agent)  # Local observation for the agent
            initial_observation = ActorRNN.process_observation(initial_observation)

            # Initialize hidden states for actor and critic networks
            input_size = actor_policies[agent].hidden_size
            actor_hidden_state = torch.zeros(1,1,  input_size)  # Shape: (1, batch_size, hidden_size)
            critic_hidden_state = torch.zeros(1,1,  input_size)
        
            action_type = random.choice([env.MOVE_UNIT, env.ATTACK_UNIT, env.FOUND_CITY, env.ASSIGN_PROJECT, env.NO_OP])

            # Placeholder for the rest of the trajectory components
            initial_action = {
                "action_type": action_type,
                "unit_id": random.randint(0, len(env.units[agent]) - 1) if env.units[agent] else 0,
                "direction": random.randint(0, 3),
                "city_id": random.randint(0, len(env.cities[agent]) - 1) if env.cities[agent] else 0,
                "project_id": random.randint(0, env.max_projects - 1)
            }
            initial_reward = 0  # No reward has been received yet
            initial_next_state = initial_state  # Initial state is also the "next state"
            initial_next_observation = initial_observation  # Same for the observation
            initial_value = 0

            # Create the initial trajectory structure
            starting_trajectory = [
                initial_state,             # State at time t
                initial_observation,       # Observation at time t
                actor_hidden_state,        # Actor hidden state
                critic_hidden_state,       # Critic hidden state
                initial_action,            # Action at time t (None at start)
                initial_reward,            # Reward at time t (0 at start)
                initial_next_state,        # Next state (same as initial state)
                initial_next_observation,   # Next observation (same as initial observation)
                initial_value
            ]
            starting_trajectories.append(starting_trajectory)

        return starting_trajectories

    def sample_action(self, action_probs):
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
        
    def flatten_observation(self, observation):
        if isinstance(observation, dict):
            # Flatten dictionary values
            tensors = []
            for key, value in observation.items():
                if isinstance(value, np.ndarray):
                    value = torch.tensor(value, dtype=torch.float32, device=self.device)
                elif not isinstance(value, torch.Tensor):
                    # handle other cases if necessary
                    pass
                tensors.append(value.flatten())
            return torch.cat(tensors).to(self.device)
        elif isinstance(observation, torch.Tensor):
            # Already a tensor, just return it as is (or flatten if needed)
            return observation.flatten().to(self.device)
        else:
            # Handle other unexpected types if necessary
            raise TypeError(f"Unsupported observation type: {type(observation)}")


    def compute_returns(self, rewards, discount=1):
        """
        Compute the discounted rewards-to-go (returns) for a given list of rewards.

        Args:
            rewards (torch.Tensor): A 1D tensor of rewards for a trajectory.
            gamma (float): Discount factor for future rewards. Default is 0.99.

        Returns:
            torch.Tensor: A 1D tensor of discounted rewards-to-go (returns).
        """
        rewards = rewards.float()  # Convert all elements to float

        returns = torch.zeros_like(rewards)

        for agent in self.env.agents:
            discounted_sum = 0.0
            # Calculate the returns in reverse order
            for t in reversed(range(len(rewards[agent]))):
                discounted_sum = float(rewards[agent, t]) + discount * discounted_sum  # Explicitly cast to float
                
                sys.stdout.flush()
                returns[agent, t] = discounted_sum

        return returns
    


#hyperparameter sweep:
wandb.init(project="hyperparameter-sweep-continuous")
sweep_config = {
'method': 'random',
'metric': {'name': 'accuracy', 'goal': 'maximize'},
'parameters': {
    'learning_rate': {
        'min': 0.0001,
        'max': 0.1,
        'distribution': 'uniform'
    },
    'batch_size': {
        'min': 16,
        'max': 128,
        'distribution': 'int_uniform'
    },
    'dropout_rate': {
        'min': 0.1,
        'max': 0.5,
        'distribution': 'uniform'
    },
    'num_hidden_layers': {
        'min': 1,
        'max': 4,
        'distribution': 'int_uniform'
    },
    'neurons_per_layer': {
        'min': 32,
        'max': 256,
        'distribution': 'int_uniform'
    }
    }
}

sweep_id = wandb.sweep(sweep_config)    
wandb.agent(sweep_id, function=ppo.train)