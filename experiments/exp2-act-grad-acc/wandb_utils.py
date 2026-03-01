import os
import wandb
from torch.utils.tensorboard.writer import SummaryWriter


def initialize_wandb_and_writer(args, run_name: str) -> SummaryWriter:
    """
    Initialize wandb and TensorBoard writer for experiment tracking.

    Args:
        args: Arguments object containing experiment configuration
        run_name: Name of the current run

    Returns:
        SummaryWriter: TensorBoard writer for logging metrics
    """
    # Initialize wandb only if API key is provided
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.init(
            project=args.wandb_project,
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

    return writer