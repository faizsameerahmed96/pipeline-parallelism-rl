import os

from torch.distributed.rpc import RRef
import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
import time

torch.backends.cudnn.deterministic = True

from network import ActorCriticNetwork

# Global optimizer and network (initialized in setup)
actor_critic_network = None
optimizer = None
max_grad_norm = 0.5

def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

def _call_method_no_grad(method, rref, *args, **kwargs):
    with torch.no_grad():
        return method(rref.local_value(), *args, **kwargs)
    
def _parameter_rrefs(module):
    return [RRef(parameter) for parameter in module.parameters()]


def backward_and_step(cnn_features, actions, old_logprobs, advantages, returns, old_values, 
                      clip_coef, vf_coef, ent_coef, norm_adv, clip_vloss):
    """
    Perform forward pass, compute loss components, backward pass, and optimizer step.
    Returns gradients w.r.t. cnn_features to send back to machine0.
    """
    global actor_critic_network, optimizer
    
    # Ensure features require gradients
    cnn_features = cnn_features.detach().requires_grad_(True)
    
    # Forward pass through actor-critic network
    _, newlogprob, entropy, newvalue = actor_critic_network.get_action_and_value(cnn_features, actions)
    
    # Compute loss components (same as machine0)
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
    optimizer.zero_grad()
    loss.backward()
    
    # Extract gradients w.r.t. input features (to send back to machine0)
    feature_grads = cnn_features.grad.clone()
    
    # Clip gradients for actor-critic parameters
    nn.utils.clip_grad_norm_(actor_critic_network.parameters(), max_grad_norm)
    
    # Optimizer step for actor-critic parameters
    optimizer.step()
    
    # Return gradients and loss components for logging
    return feature_grads, pg_loss.item(), v_loss.item(), entropy_loss.item()


def setup():
    global actor_critic_network, optimizer
    
    rank = int(os.environ["RANK"])
    rpc.init_rpc(
        name=f"worker{rank}", rank=rank, world_size=int(os.environ["WORLD_SIZE"])
    )
    print("RPC initialized successfully.", flush=True)
    
    # Initialize actor-critic network and optimizer on machine1
    actor_critic_network = ActorCriticNetwork()
    optimizer = torch.optim.Adam(actor_critic_network.parameters(), lr=1e-5)
    print("ActorCriticNetwork and optimizer initialized on machine1.", flush=True)
    
    # Keep RPC running (don't shutdown)
    rpc.shutdown()


if __name__ == "__main__":
    setup()
