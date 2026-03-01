#!/usr/bin/env python3
"""
Simple checkpoint evaluation script.
Runs a checkpoint on CarRacing environment n times and saves results.
"""

import os
import json
import random
import time
from dataclasses import dataclass
from typing import List

import gymnasium as gym
import numpy as np
import torch
import tyro

from env import make_env
from network import ActorCriticNetwork, CNNNetwork


@dataclass
class EvalArgs:
    """Arguments for checkpoint evaluation."""
    cnn_checkpoint_path: str
    actor_critic_checkpoint_path: str
    num_episodes: int = 10
    seed: int = 42
    cuda: bool = False
    env_id: str = "CarRacing-v3"





def evaluate_checkpoint(args: EvalArgs) -> List[float]:
    """
    Evaluate a checkpoint for multiple episodes.
    
    Returns:
        List of episodic returns
    """
    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Create evaluation environment
    env = make_env(args.env_id, 0, False, "eval", 0.99)()
    
    # Load CNN network
    cnn_network = CNNNetwork(
        type('MockEnv', (), {'single_observation_space': env.observation_space})(),
        learning_rate=0.00004  # Not used during evaluation
    ).to(device)
    
    cnn_checkpoint = torch.load(args.cnn_checkpoint_path, map_location=device)
    cnn_network.load_state_dict(cnn_checkpoint['model_state_dict'])
    cnn_network.eval()
    
    # Load ActorCritic network locally
    actor_critic_network = ActorCriticNetwork().to(device)
    actor_critic_network.load_model(args.actor_critic_checkpoint_path)
    actor_critic_network.eval()
    
    episodic_returns = []
    
    print(f"Starting evaluation for {args.num_episodes} episodes...")
    
    for episode in range(args.num_episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        obs = torch.Tensor(obs).unsqueeze(0).to(device)
        done = False
        episode_return = 0.0
        
        while not done:
            with torch.no_grad():
                # Get CNN features
                cnn_features = cnn_network(obs)
                
                # Get action from local ActorCritic
                action, _, _ = actor_critic_network.get_action_and_value(
                    cnn_features,
                    no_grad=True
                )
                
                # Execute action
                obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
                obs = torch.Tensor(obs).unsqueeze(0).to(device)
                done = terminated or truncated
                episode_return += reward
        
        episodic_returns.append(episode_return)
        print(f"Episode {episode + 1}/{args.num_episodes}: {episode_return:.2f}")
    
    env.close()
    return episodic_returns


def save_evaluation_results(args: EvalArgs, returns: List[float]):
    """Save evaluation results to JSON file in checkpoint directory."""
    # Get checkpoint directory (assume both checkpoints are in same run folder)
    checkpoint_dir = os.path.dirname(os.path.dirname(args.cnn_checkpoint_path))
    
    # Create results data
    results = {
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cnn_checkpoint_path": args.cnn_checkpoint_path,
        "actor_critic_checkpoint_path": args.actor_critic_checkpoint_path,
        "num_episodes": args.num_episodes,
        "seed": args.seed,
        "env_id": args.env_id,
        "episodic_returns": returns,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "median_return": float(np.median(returns)),
        "q1_return": float(np.percentile(returns, 25)),
        "q3_return": float(np.percentile(returns, 75))
    }
    
    # Save results
    results_file = os.path.join(checkpoint_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Mean return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"Median return: {results['median_return']:.2f}")
    print(f"Range: [{results['min_return']:.2f}, {results['max_return']:.2f}]")
    print(f"Quartiles: Q1={results['q1_return']:.2f}, Q3={results['q3_return']:.2f}")


def main():
    """Main evaluation function."""
    args = tyro.cli(EvalArgs)
    
    # Verify checkpoint files exist
    if not os.path.exists(args.cnn_checkpoint_path):
        raise FileNotFoundError(f"CNN checkpoint not found: {args.cnn_checkpoint_path}")
    if not os.path.exists(args.actor_critic_checkpoint_path):
        raise FileNotFoundError(f"ActorCritic checkpoint not found: {args.actor_critic_checkpoint_path}")
    
    print("Starting checkpoint evaluation...")
    returns = evaluate_checkpoint(args)
    
    print("Saving results...")
    save_evaluation_results(args, returns)
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main()