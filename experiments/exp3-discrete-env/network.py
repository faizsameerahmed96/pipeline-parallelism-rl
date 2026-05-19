import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
import os
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

        actor_output_dim = 9  # Enduro action space
        self.actor = nn.Sequential(
            nn.Linear(cnn_output_size, 512),
            nn.ReLU(),
            layer_init(nn.Linear(512, actor_output_dim), std=0.01),
        )
        
        # Initialize optimizer for this network
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        self.max_grad_norm = 0.5
        
        # Initialize global feature gradient accumulator
        self.global_feature_grads = None

        # Running stats for surprise-based filtering
        self.running_mean = None
        self.running_var = None

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        # with torch.no_grad():
        #     sample_input = torch.zeros(1, *envs.single_observation_space.shape)
        #     cnn_output_size = self.cnn(sample_input).shape[1]

    def get_global_feature_grads(self):
        """Return a copy of the gradient accumulation buffer."""
        if self.global_feature_grads is None:
            return None
        return self.global_feature_grads.clone()

    def get_action_and_value(self, features, action=None, no_grad=False):
        if no_grad:
            with torch.no_grad():
                return self._get_action_and_value(features, action)
        else:
            return self._get_action_and_value(features, action)

    def _get_action_and_value(self, features, action=None):
        logits = self.actor(features)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, self.critic(features)

    def get_value(self, cnn_features, no_grad=False):
        if no_grad:
            with torch.no_grad():
                return self.critic(cnn_features)
        else:
            return self.critic(cnn_features)
    
    def backward_and_step(self, cnn_features, actions, old_logprobs, advantages, returns, old_values,
                          clip_coef, vf_coef, ent_coef, norm_adv, clip_vloss, gradient_stats=False, accumulated_grads=False, accumulate_grads_percentile=0.90, decay_buffer=1.0,
                          surprise_compress=False, surprise_compress_percentile=0.90, surprise_compress_ema_alpha=0.5,
                          surprise_sync_interval=10, surprise_minibatch_counter=0, surprise_warmup=False):
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
        relevant_grads_from_global_feature_grads = None # This will only contain the grads that are only over a certain percentile
        buffer_stats = None

        # During warm start for surprise technique: accumulate running stats without compressing
        if surprise_warmup:
            grad_per_dim = feature_grads.mean(dim=0)  # (4096,)
            if self.running_mean is None:
                self.running_mean = grad_per_dim.clone()
                self.running_var = torch.zeros_like(grad_per_dim)
            else:
                alpha = surprise_compress_ema_alpha
                delta = grad_per_dim - self.running_mean
                self.running_mean = alpha * self.running_mean + (1 - alpha) * grad_per_dim
                self.running_var = alpha * self.running_var + (1 - alpha) * delta * (grad_per_dim - self.running_mean)
                self.running_var = torch.clamp(self.running_var, min=0)

        if accumulated_grads:
            # Accumulate gradients
            if self.global_feature_grads is None:
                self.global_feature_grads = torch.zeros_like(feature_grads)

            # Decay and accumulate
            self.global_feature_grads = self.global_feature_grads * decay_buffer + feature_grads

            # Find 90th percentile of absolute gradient values from accumulated gradients
            abs_grads = torch.abs(self.global_feature_grads)
            grads_above_accumulate_grads_percentile = torch.quantile(abs_grads, accumulate_grads_percentile)

            # Buffer distribution stats for logging
            abs_flat = abs_grads.flatten()
            buffer_stats = {
                'mean': abs_flat.mean().item(),
                'p10': torch.quantile(abs_flat, 0.10).item(),
                'p25': torch.quantile(abs_flat, 0.25).item(),
                'p50': torch.quantile(abs_flat, 0.50).item(),
                'p90': torch.quantile(abs_flat, 0.90).item(),
                'p99': torch.quantile(abs_flat, 0.99).item(),
                'threshold': grads_above_accumulate_grads_percentile.item(),
            }

            # Create mask for values above 90th percentile
            mask = abs_grads >= grads_above_accumulate_grads_percentile
            
            # Get indices where mask is True
            indices = torch.nonzero(mask, as_tuple=False)  # Shape: (num_above_threshold, 2)
            
            # Get the gradient values at those indices
            values = self.global_feature_grads[mask]  # Shape: (num_above_threshold,)
            
            # Format: [num_elements, indices (row, col pairs), values]
            relevant_grads_from_global_feature_grads = {
                'indices': indices.short(),  # Convert to int16 to save space
                'values': values,
                'shape': feature_grads.shape
            }
            
            # Zero out the selected gradients from global accumulator
            self.global_feature_grads[mask] = 0

            feature_grads = None

        elif surprise_compress:
            # Per-dimension running stats (4096,) applied element-wise via broadcast
            grad_per_dim = feature_grads.mean(dim=0)  # (4096,)

            # Compute z-scores: z[i,j] = |grad[i,j] - μ[j]| / σ[j]
            running_std = torch.sqrt(self.running_var + 1e-8)  # (4096,)
            z_scores = torch.abs(feature_grads - self.running_mean) / running_std  # (minibatch, 4096)

            # Log z-score distribution
            z_flat = z_scores.flatten()
            buffer_stats = {
                'z_p25': torch.quantile(z_flat, 0.25).item(),
                'z_p50': torch.quantile(z_flat, 0.50).item(),
                'z_p90': torch.quantile(z_flat, 0.90).item(),
                'z_p99': torch.quantile(z_flat, 0.99).item(),
            }

            # Select top N% most surprising elements from full (minibatch, 4096) tensor
            z_threshold = torch.quantile(z_flat, surprise_compress_percentile)
            mask = z_scores >= z_threshold

            # Sparse format: indices + values
            indices = torch.nonzero(mask, as_tuple=False)  # (K, 2)
            values = feature_grads[mask]  # (K,)

            relevant_grads_from_global_feature_grads = {
                'indices': indices.short(),
                'values': values,
                'shape': feature_grads.shape,
                'technique': 'surprise',
            }

            # Sync stats periodically (first compressed minibatch always syncs)
            if surprise_minibatch_counter % surprise_sync_interval == 0:
                relevant_grads_from_global_feature_grads['running_mean'] = self.running_mean.clone()
                relevant_grads_from_global_feature_grads['running_std'] = running_std.clone()

            # Update running stats with EMA
            alpha = surprise_compress_ema_alpha
            delta = grad_per_dim - self.running_mean
            self.running_mean = alpha * self.running_mean + (1 - alpha) * grad_per_dim
            self.running_var = alpha * self.running_var + (1 - alpha) * delta * (grad_per_dim - self.running_mean)
            self.running_var = torch.clamp(self.running_var, min=0)

            feature_grads = None

        elif gradient_stats:
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
        return feature_grads, feature_grads_stats, relevant_grads_from_global_feature_grads, pg_loss.item(), v_loss.item(), entropy_loss.item(), buffer_stats
    
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
