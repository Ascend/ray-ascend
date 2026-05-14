# HCCL Collective Communication

> _Last updated: 03/24/2026_

ray-ascend provides HCCL (Huawei Collective Communication Library) support for
distributed collective operations across Ray actors.

## Available Collective Operations

- **broadcast**: Send data from one rank to all ranks
- **allreduce**: Combine data from all ranks and distribute the result
- **allgather**: Gather data from all ranks to each rank
- **reduce**: Combine data from all ranks to one rank
- **reducescatter**: Combine data and scatter the result to all ranks
- **send/recv**: Point-to-point communication
- **barrier**: Synchronize all ranks

## Quick Example: HCCL Collective Group

```python
import ray
from ray.util import collective
from ray_ascend.collective.hccl_collective_group import HCCLGroup

# Initialize Ray
ray.init()

# Register the HCCL backend
ray.register_collective_backend("HCCL", HCCLGroup)

# Create actors with NPU resources
@ray.remote(resources={"NPU": 1})
class Worker:
    def __init__(self):
        import torch
        import torch_npu
        self.device = torch.npu.current_device()

    def setup_group(self, world_size, rank, group_name):
        self.group = HCCLGroup(world_size, rank, group_name)

    def do_allreduce(self, data):
        import torch
        tensor = torch.tensor(data, dtype=torch.float32).npu()
        self.group.allreduce(tensor)
        return tensor.cpu().tolist()

    def destroy(self):
        self.group.destroy_group()

# Create workers
world_size = 2
actors = [Worker.remote() for _ in range(world_size)]

# Create collective group
collective.create_collective_group(
    actors,
    world_size,
    list(range(world_size)),
    backend="HCCL",
    group_name="my_hccl_group",
)

# Perform allreduce
results = ray.get([
    actors[i].do_allreduce.remote([1.0 * (i + 1), 2.0 * (i + 1)])
    for i in range(world_size)
])
print("Allreduce results:", results)  # Both should show [3.0, 6.0]

# Cleanup
ray.get([actor.destroy.remote() for actor in actors])
ray.shutdown()
```

## Using Ray's Collective API

You can also use Ray's high-level collective API:

```python
import ray
from ray.util import collective
from ray_ascend.collective.hccl_collective_group import HCCLGroup

ray.init()
ray.register_collective_backend("HCCL", HCCLGroup)

@ray.remote(resources={"NPU": 1})
class Worker:
    def broadcast_tensor(self, src_rank=0):
        import torch
        tensor = torch.ones(10).npu() if self.rank == src_rank else torch.zeros(10).npu()
        collective.broadcast(tensor, src_rank=src_rank, group_name="my_hccl_group")
        return tensor.cpu().tolist()

# Create and setup group...

# Each actor broadcasts in SPMD manner
results = ray.get([actor.broadcast_tensor.remote() for actor in actors])
```

## Supported Tensor Types

HCCL supports common PyTorch types:

- `int8`, `int16`, `int32`, `int64`
- `uint8`, `uint16`, `uint32`, `uint64`
- `float16`, `float32`, `float64`
- `bfloat16`

## Supported Reduce Operations

- `SUM`
- `PRODUCT`
- `MAX`
- `MIN`
