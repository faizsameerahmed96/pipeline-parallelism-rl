import tyro
from dataclasses import dataclass


@dataclass
class Args:
    seed: int = 1
    cuda: bool = False
    env_id: str = "CarRacing-v3"

    # Algorithm specific arguments
    total_timesteps: int = 1_000_000
    learning_rate: float = 0.00004
    num_envs: int = 8
    num_steps: int = 4096
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

    # Model saving/loading args
    save_model_freq: int | None = 10
    cnn_network_checkpoint_path: str | None = None
    agent_network_checkpoint_path: str | None = None

    # Wandb args
    wandb_project: str = "data-paralellism-rl-2026-02"

    # Compression related args
    gradient_compression_technique: str | None = 'accumulate-grads' # 'stats', 'accumulate-grads'
    accumulate_grads_percentile: float | None = 0.90
    decay_buffer: float = 0.95 # Decay factor for accumulated gradients buffer (1.0 = no decay)
    warm_start_steps: int = 30_000 # Number of steps before starting compression

    # to be filled in runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 25


def get_args() -> Args:
    """Parse command line arguments and compute derived values."""
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    return args