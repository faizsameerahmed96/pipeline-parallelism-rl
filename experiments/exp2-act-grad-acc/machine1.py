import os

from torch.distributed.rpc import RRef
import torch
import torch.distributed.rpc as rpc
import time

torch.backends.cudnn.deterministic = True

def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

def _call_method_no_grad(method, rref, *args, **kwargs):
    with torch.no_grad():
        return method(rref.local_value(), *args, **kwargs)
    
def _parameter_rrefs(module):
    return [RRef(parameter) for parameter in module.parameters()]

def setup():
    rank = int(os.environ["RANK"])
    rpc.init_rpc(
        name=f"worker{rank}", rank=rank, world_size=int(os.environ["WORLD_SIZE"])
    )
    print("RPC initialized successfully.", flush=True)

    rpc.shutdown()


if __name__ == "__main__":
    setup()
