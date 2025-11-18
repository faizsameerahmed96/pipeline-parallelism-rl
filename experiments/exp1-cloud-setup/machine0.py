import os
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.nn as nn
import tyro
import wandb
from torch.utils.tensorboard.writer import SummaryWriter

from env import make_env
from network import ActorCriticNetwork, CNNNetwork
from torch.distributed.autograd import context as dist_autograd_context
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef


@dataclass
class Args:
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = False
    env_id: str = "CarRacing-v3"
    technique: str = "cloud-setup"

    # Algorithm specific arguments
    total_timesteps: int = 500_000
    learning_rate: float = 0.00005
    num_envs: int = 1
    num_steps: int = 4096
    anneal_lr: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 32
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    owner = rref.owner()
    return rpc.rpc_sync(owner, _call_method, args=(method, rref, *args), kwargs=kwargs)


def _call_method_no_grad(method, rref, *args, **kwargs):
    with torch.no_grad():
        return method(rref.local_value(), *args, **kwargs)


def _parameter_rrefs(module):
    return [RRef(parameter) for parameter in module.parameters()]


def _remote_method_no_grad(method, rref, *args, **kwargs):
    owner = rref.owner()
    return rpc.rpc_sync(
        owner, _call_method_no_grad, args=(method, rref, *args), kwargs=kwargs
    )


def main():
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    run_name = f"{int(time.time())}-{args.technique}"
    print(f"Run: {run_name}")
    
    # Initialize wandb only if API key is provided
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.init(
            project="data-paralellism-rl",
            entity=None,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
            dir="/workspace/runs",
        )
        # Create TensorBoard writer in wandb's run directory for automatic syncing
        writer = SummaryWriter(f"/workspace/runs/{run_name}")
        print("wandb initialized successfully", flush=True)
    else:
        writer = SummaryWriter(f"/workspace/runs/{run_name}")
        print("WANDB_API_KEY not found, skipping wandb initialization", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, i, False, run_name, args.gamma)
            for i in range(args.num_envs)
        ]
    )

    agent = CNNNetwork(envs).to(device)

    remote_agent_rref = rpc.remote("worker1", ActorCriticNetwork)
    print(f"Remote reference to worker1 obtained.", flush=True)

    remote_param_rrefs = _remote_method(_parameter_rrefs, remote_agent_rref)
    local_param_rrefs = [RRef(parameter) for parameter in agent.parameters()]
    optimizer = DistributedOptimizer(
        torch.optim.Adam,
        remote_param_rrefs + local_param_rrefs,
        lr=1e-5,
    )

    # store information from rollout
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        print(f"Iteration {iteration}/{args.num_iterations}", flush=True)
        # Rollout
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                cnn_features = agent(next_obs)
                action, logprob, _, value = _remote_method_no_grad(
                    ActorCriticNetwork.get_action_and_value,
                    remote_agent_rref,
                    cnn_features,
                )
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

            if "episode" in infos:
                ep_infos = infos["episode"]
                print(
                    f"global_step={global_step}, episodic_return={ep_infos['r']}",
                    flush=True,
                )
                writer.add_scalar(
                    "charts/episodic_return", ep_infos["r"][0], global_step
                )

        # bootstrap value if not done
        with torch.no_grad():
            cnn_features = agent(next_obs)
            next_value = _remote_method_no_grad(
                ActorCriticNetwork.get_value, remote_agent_rref, cnn_features
            )
            next_value = next_value.reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                with dist_autograd_context() as context_id:
                    cnn_features = agent(b_obs[mb_inds])
                    _, newlogprob, entropy, newvalue = _remote_method(
                        ActorCriticNetwork.get_action_and_value,
                        remote_agent_rref,
                        cnn_features,
                        b_actions[mb_inds],
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    # with torch.no_grad():
                    #     # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    #     old_approx_kl = (-logratio).mean()
                    #     approx_kl = ((ratio - 1) - logratio).mean()
                    #     clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    )

                    dist_autograd.backward(context_id, [loss])

                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step(context_id)

            # if args.target_kl is not None and approx_kl > args.target_kl:
            #     break
        
        # Log training metrics
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        time_elapsed = time.time() - start_time
        print(f"Time elapsed: {time_elapsed:.2f}s", flush=True)
        writer.add_scalar("charts/time_elapsed", time_elapsed, iteration)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/learning_rate", args.learning_rate, global_step)
        sps = int(global_step / time_elapsed)
        print(f"SPS: {sps}")
        writer.add_scalar("charts/SPS", sps, global_step)

    writer.close()
    print("Machine shutting down", flush=True)
    time.sleep(1000)


def setup():
    rank = int(os.environ["RANK"])
    rpc.init_rpc(
        name=f"worker{rank}", rank=rank, world_size=int(os.environ["WORLD_SIZE"])
    )
    print("RPC initialized successfully.", flush=True)


if __name__ == "__main__":
    print("Machine 0 running!", flush=True)
    setup()
    main()
