import os
import random

from torch.distributed.rpc import RRef
import torch
import torch.distributed.rpc as rpc
import time
import numpy as np

# Enable deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

def _call_method_no_grad(method, rref, *args, **kwargs):
    with torch.no_grad():
        return method(rref.local_value(), *args, **kwargs)
    
def _parameter_rrefs(module):
    return [RRef(parameter) for parameter in module.parameters()]

def setup():
    rank = int(os.environ["RANK"])
    
    # Set seeds for deterministic behavior
    seed = 1  # Using same seed as machine0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    
    rpc.init_rpc(
        name=f"worker{rank}", rank=rank, world_size=int(os.environ["WORLD_SIZE"])
    )
    print("RPC initialized successfully.", flush=True)

    rpc.shutdown()


if __name__ == "__main__":
    setup()
