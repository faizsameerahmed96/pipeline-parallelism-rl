import torch.nn as nn
import numpy as np
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.distributions.normal import Normal
import os
import torch
import torch


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CNNNetwork(nn.Module):
    def __init__(self, envs, learning_rate=0.00005):
        super().__init__()

        n_input_channels = envs.single_observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Initialize optimizer for this network
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # with torch.no_grad():
        #     sample_input = torch.zeros(1, *envs.single_observation_space.shape)
        #     cnn_output_size = self.cnn(sample_input).shape[1]

    def forward(self, obs):
        features = self.cnn(obs)
        return features
    
    def save_model(self, checkpoint_dir, iteration, args=None):
        """Save the CNNNetwork model checkpoint."""
        save_path = f"{checkpoint_dir}cnn/"
        os.makedirs(save_path, exist_ok=True)
        
        checkpoint_path = f"{save_path}iteration_{iteration}.pt"
        checkpoint_data = {
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if args is not None:
            checkpoint_data['args'] = vars(args)
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"CNNNetwork saved to {checkpoint_path}", flush=True)
    
    def load_model(self, checkpoint_path):
        """Load the CNNNetwork model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"CNNNetwork and optimizer loaded from {checkpoint_path}", flush=True)

        return checkpoint.get('iteration', 0)


class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        cnn_output_size = 4096

        self.critic = nn.Sequential(
            nn.Linear(cnn_output_size, 512),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

        actor_output_dim = int(np.prod((3,)))  # car racing action space
        self.actor_mean = nn.Sequential(
            nn.Linear(cnn_output_size, 512),
            nn.ReLU(),
            layer_init(nn.Linear(512, actor_output_dim), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, actor_output_dim))
        
        # Initialize optimizer for this network
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        self.max_grad_norm = 0.5

        # with torch.no_grad():
        #     sample_input = torch.zeros(1, *envs.single_observation_space.shape)
        #     cnn_output_size = self.cnn(sample_input).shape[1]

    def get_action_and_value(self, features, action=None, no_grad=False):
        if no_grad:
            with torch.no_grad():
                return self._get_action_and_value(features, action)
        else:
            return self._get_action_and_value(features, action)

    def _get_action_and_value(self, features, action=None):
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)

        LOG_STD_MAX = 2
        LOG_STD_MIN = -5
        action_logstd = torch.clamp(action_logstd, LOG_STD_MIN, LOG_STD_MAX)
        action_std = torch.exp(action_logstd)

        base_dist = Normal(action_mean, action_std)
        squashed_dist = TransformedDistribution(
            base_dist, TanhTransform()
        )  # ensure the distribution is within -1,1

        if action is None:
            action = squashed_dist.sample()

        log_prob = squashed_dist.log_prob(action).sum(1)

        # The entropy is harder to compute for a transformed distribution,
        # so we can approximate it with the entropy of the base distribution.
        entropy_2 = base_dist.entropy().sum(1)

        return action, log_prob, entropy_2, self.critic(features)

    def get_value(self, cnn_features, no_grad=False):
        if no_grad:
            with torch.no_grad():
                return self.critic(cnn_features)
        else:
            return self.critic(cnn_features)
    
    def backward_and_step(self, cnn_features, actions, old_logprobs, advantages, returns, old_values, 
                          clip_coef, vf_coef, ent_coef, norm_adv, clip_vloss, gradient_stats=False):
        """
        Perform forward pass, compute loss components, backward pass, and optimizer step.
        Returns gradients w.r.t. cnn_features to send back to machine0.
        
        Args:
            gradient_stats: If True, returns mean and std of gradients (2, 4096) instead of full gradients (batch_size, 4096)
        """
        # Ensure features require gradients
        cnn_features = cnn_features.detach().requires_grad_(True)

        # Forward pass through actor-critic network
        _, newlogprob, entropy, newvalue = self.get_action_and_value(cnn_features, actions)
        
        # Compute loss components
        logratio = newlogprob - old_logprobs
        ratio = logratio.exp()
        
        # Normalize advantages if needed
        if norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        
        # Value loss
        newvalue = newvalue.view(-1)
        if clip_vloss:
            v_loss_unclipped = (newvalue - returns) ** 2
            v_clipped = old_values + torch.clamp(
                newvalue - old_values,
                -clip_coef,
                clip_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - returns) ** 2).mean()
        
        entropy_loss = entropy.mean()
        loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Extract gradients w.r.t. input features (to send back to machine0)
        feature_grads = cnn_features.grad.clone()

        feature_grads_stats = None
        
        if gradient_stats:
            # Compute mean and std across the batch dimension (dim=0)
            grad_mean = feature_grads.mean(dim=0, keepdim=True)
            grad_std = feature_grads.std(dim=0, keepdim=True)
            feature_grads_stats = torch.cat([grad_mean, grad_std], dim=0)
            feature_grads = None
        
        # Clip gradients for actor-critic parameters
        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        
        # Optimizer step for actor-critic parameters
        self.optimizer.step()
        
        # Return gradients and loss components for logging
        return feature_grads, feature_grads_stats, pg_loss.item(), v_loss.item(), entropy_loss.item()
    
    def save_model(self, checkpoint_dir, iteration):
        """Save the ActorCriticNetwork model checkpoint."""
        save_path = f"{checkpoint_dir}actor_critic/"
        os.makedirs(save_path, exist_ok=True)
        
        checkpoint_path = f"{save_path}iteration_{iteration}.pt"
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, checkpoint_path)
        
        print(f"ActorCriticNetwork saved to {checkpoint_path}", flush=True)
    
    def load_model(self, checkpoint_path):
        """Load the ActorCriticNetwork model checkpoint."""
        
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"ActorCriticNetwork loaded from {checkpoint_path}", flush=True)
        return checkpoint.get('iteration', 0)
