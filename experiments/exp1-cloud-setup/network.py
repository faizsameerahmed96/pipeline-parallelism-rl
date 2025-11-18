import torch.nn as nn
import numpy as np
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.distributions.normal import Normal
import torch

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CNNNetwork(nn.Module):
    def __init__(self, envs):
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

        # with torch.no_grad():
        #     sample_input = torch.zeros(1, *envs.single_observation_space.shape)
        #     cnn_output_size = self.cnn(sample_input).shape[1]
        

    def forward(self, obs):
        features = self.cnn(obs)
        return features

class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        cnn_output_size = 4096
        
        self.critic = nn.Sequential(
            nn.Linear(cnn_output_size, 512),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0)
        )

        actor_output_dim = int(np.prod((3,))) # car racing action space
        self.actor_mean = nn.Sequential(
            nn.Linear(cnn_output_size, 512),
            nn.ReLU(),
            layer_init(nn.Linear(512, actor_output_dim), std=0.01)
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, actor_output_dim))

        # with torch.no_grad():
        #     sample_input = torch.zeros(1, *envs.single_observation_space.shape)
        #     cnn_output_size = self.cnn(sample_input).shape[1]

    def get_action_and_value(self, features, action=None):
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)

        LOG_STD_MAX = 2
        LOG_STD_MIN = -5
        action_logstd = torch.clamp(action_logstd, LOG_STD_MIN, LOG_STD_MAX)
        action_std = torch.exp(action_logstd)

        base_dist = Normal(action_mean, action_std)
        squashed_dist = TransformedDistribution(base_dist, TanhTransform()) # ensure the distribution is within -1,1

        if action is None:
            action = squashed_dist.sample()

        log_prob = squashed_dist.log_prob(action).sum(1)
        
        # The entropy is harder to compute for a transformed distribution,
        # so we can approximate it with the entropy of the base distribution.
        entropy_2 = base_dist.entropy().sum(1)
            
        return action, log_prob, entropy_2, self.critic(features)
    

    def get_value(self, cnn_features):
        return self.critic(cnn_features)