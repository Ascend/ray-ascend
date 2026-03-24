# User Guide

> _Last updated: 03/24/2026_

## Target

This guide is intended for end users and integrators who want to use the features of
ray-ascend. It provides step-by-step instructions for installation, configuration, and
usage of both HCCL collective communication and YuanRong direct tensor transport.

## Prerequisites

- **Architecture**: aarch64, x86
- **OS Kernel**: Linux
- **Python**: >= 3.10, \<= 3.11
- **Ray**: Same version as ray-ascend

Optional dependencies for specific features:

- **CANN == 8.2.rc1**: Required for NPU features (HCCL, NPU tensor transport)
- **torch == 2.7.1, torch-npu == 2.7.1.post1**: Required for PyTorch NPU support

## Installation

### Basic Installation (HCCL Only)

Install the base package with HCCL collective communication support:

```bash
pip install ray-ascend
```

### With YuanRong Direct Transport Support

Install with YuanRong (YR) direct tensor transport support:

```bash
pip install "ray-ascend[yr]"
```

### From Source (Editable Installation)

For development or to use the latest version:

```bash
git clone https://github.com/Ascend/ray-ascend.git
cd ray-ascend
pip install -e ".[all]"
```

## CANN Setup (for NPU Features)

If you have Ascend NPU devices and want to use HCCL or NPU tensor transport, you need to
install the CANN toolkit.

### Using CANN Docker (Recommended)

We recommend using the official CANN Docker images for the easiest setup:

```bash
# For Ascend NPU A3
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.2.rc1-a3-ubuntu22.04-py3.11

# For Ascend NPU 910B
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.2.rc1-910b-ubuntu22.04-py3.11
```

For more details on running the container, see the
[official CANN image documentation](https://github.com/Ascend/cann-container-image).

## HCCL Collective Communication

ray-ascend provides HCCL (Huawei Collective Communication Library) support for
distributed collective operations across Ray actors.

### Available Collective Operations

- **broadcast**: Send data from one rank to all ranks
- **allreduce**: Combine data from all ranks and distribute the result
- **allgather**: Gather data from all ranks to each rank
- **reduce**: Combine data from all ranks to one rank
- **reducescatter**: Combine data and scatter the result to all ranks
- **send/recv**: Point-to-point communication
- **barrier**: Synchronize all ranks

### Quick Example: HCCL Collective Group

```python
import ray
from ray.util import collective
from ray_ascend.collective import HCCLGroup

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

### Using Ray's Collective API

You can also use Ray's high-level collective API:

```python
import ray
from ray.util import collective
from ray_ascend.collective import HCCLGroup

ray.init()
ray.register_collective_backend("HCCL", HCCLGroup)

@ray.remote(resources={"NPU": 1})
class Worker:
    def broadcast_tensor(self, src_rank=0):
        import torch
        tensor = torch.ones(10).npu() if self.rank == src_rank else torch.zeros(10).npu()
        collective.broadcast(tensor, src_rank=src_rank, group_name="my_group")
        return tensor.cpu().tolist()

# Create and setup group...

# Each actor broadcasts in SPMD manner
results = ray.get([actor.broadcast_tensor.remote() for actor in actors])
```

### Collective API Reference

#### `HCCLGroup`

The main class for HCCL collective communication.

**Constructor:**

```python
HCCLGroup(world_size: int, rank: int, group_name: str)
```

**Methods:**

- `broadcast(tensor, broadcast_options)`: Broadcast tensor from root rank
- `allreduce(tensor, allreduce_options)`: All-reduce tensor across group
- `allgather(tensor_list, tensor, allgather_options)`: Gather tensors from all ranks
- `reduce(tensor, reduce_options)`: Reduce tensor to root rank
- `reducescatter(tensor, tensor_list, reducescatter_options)`: Reduce and scatter
- `send(tensor, send_options)`: Send tensor to peer
- `recv(tensor, recv_options)`: Receive tensor from peer
- `barrier(barrier_options)`: Synchronize all ranks
- `destroy_group()`: Clean up communicator resources

## YuanRong Direct Tensor Transport

YuanRong (YR) direct transport enables efficient zero-copy transfer of both CPU and NPU
tensors between Ray actors.

### Features

- **Zero-copy transfer**: Efficient memory sharing via OpenYuanRong DataSystem
- **Dual device support**: Works with both CPU and NPU tensors
- **One-sided communication**: Pull-based model for efficient transfers
- **Automatic garbage collection**: Cleans up resources when no longer needed

### Prerequisites for YR Transport

1. **Install ray-ascend with YR support:**

    ```bash
    pip install "ray-ascend[yr]"
    ```

1. **Install etcd** (required for YR DataSystem cluster coordination):

    ```bash
    ETCD_VERSION="v3.6.5"
    ARCH="linux-arm64"  # or "linux-amd64" for x86

    tar -xvf etcd-${ETCD_VERSION}-${ARCH}.tar.gz
    sudo ln -sf "$(pwd)/etcd-${ETCD_VERSION}-${ARCH}/etcd" /usr/local/bin/etcd
    sudo ln -sf "$(pwd)/etcd-${ETCD_VERSION}-${ARCH}/etcdctl" /usr/local/bin/etcdctl
    ```

1. **Set environment variables** for the YR DataSystem worker:

    ```bash
    export YR_DS_WORKER_HOST="127.0.0.1"
    export YR_DS_WORKER_PORT="31502"
    ```

### Quick Example: YR Tensor Transport

```python
import ray
from ray.experimental import register_tensor_transport
from ray_ascend.direct_transport import YRTensorTransport

# Initialize Ray
ray.init()

# Register YR tensor transport for both CPU and NPU tensors
register_tensor_transport("YR", ["npu", "cpu"], YRTensorTransport)

@ray.remote(resources={"NPU": 1})
class NPUActor:
    @ray.method(tensor_transport="YR")
    def get_npu_tensor(self):
        """Return an NPU tensor via YR transport."""
        import torch
        import torch_npu
        return torch.randn(1024, 1024, device="npu")

    @ray.method(tensor_transport="YR")
    def process_npu_tensor(self, tensor):
        """Receive an NPU tensor via YR transport and process it."""
        return tensor.sum().item()

@ray.remote
class CPUActor:
    @ray.method(tensor_transport="YR")
    def get_cpu_tensor(self):
        """Return a CPU tensor via YR transport."""
        import torch
        return torch.randn(1024, 1024)

# Create actors
npu_sender = NPUActor.remote()
npu_receiver = NPUActor.remote()
cpu_worker = CPUActor.remote()

# Transfer NPU tensor via HCCS
npu_tensor = ray.get(npu_sender.get_npu_tensor.remote())
result = ray.get(npu_receiver.process_npu_tensor.remote(npu_tensor))
print("NPU tensor sum:", result)

# Transfer CPU tensor via RDMA (if available)
cpu_tensor = ray.get(cpu_worker.get_cpu_tensor.remote())
print("CPU tensor shape:", cpu_tensor.shape)

ray.shutdown()
```

### Combined HCCL + YR Transport

For advanced use cases, you can use both HCCL collective communication and YR direct
transport together:

```python
import ray
from ray.util.collective import create_collective_group
from ray.experimental import register_tensor_transport
from ray_ascend.collective import HCCLGroup
from ray_ascend.direct_transport import YRTensorTransport

# Register both backends
ray.register_collective_backend("HCCL", HCCLGroup)
register_tensor_transport("YR", ["npu", "cpu"], YRTensorTransport)

@ray.remote(resources={"NPU": 1})
class RayActor:
    def __init__(self):
        import torch
        import torch_npu

    @ray.method(tensor_transport="YR")
    def random_tensor(self):
        """Return an NPU tensor via YR transport."""
        import torch
        return torch.zeros(1024, device="npu")

    def sum(self, tensor):
        """Process a received tensor."""
        return tensor.sum()

# Create actors
sender, receiver = RayActor.remote(), RayActor.remote()

# Create HCCL collective group
group = create_collective_group([sender, receiver], backend="HCCL")

# Use YR transport for tensor transfer
tensor = sender.random_tensor.remote()
result = receiver.sum.remote(tensor)
print(ray.get(result))
```

### YR Transport API Reference

#### `YRTensorTransport`

The main class for YuanRong direct tensor transport.

**Constructor:**

```python
YRTensorTransport()
```

**Environment Variables:**

- `YR_DS_WORKER_HOST`: Host address of the YR DataSystem worker (default: none,
  required)
- `YR_DS_WORKER_PORT`: Port of the YR DataSystem worker (default: none, required)

**Methods:**

- `tensor_transport_backend()`: Returns "YR"
- `is_one_sided()`: Returns True (one-sided communication)
- `get_ds_client(device_type)`: Get or create the DataSystem client
- `actor_has_tensor_transport(actor)`: Check if actor has YR transport available
- `extract_tensor_transport_metadata(obj_id, gpu_object)`: Extract metadata for
  transport
- `recv_multiple_tensors(obj_id, tensor_transport_metadata, communicator_metadata)`:
  Receive tensors
- `garbage_collect(obj_id, tensor_transport_meta)`: Clean up resources

## Best Practices

### HCCL Collective Communication

1. **Device Consistency**: Ensure all tensors used in collective operations reside on
   the same NPU device that was used during communicator initialization.

1. **Group Cleanup**: Always call `destroy_group()` when done to free communicator
   resources.

1. **Rank Coordination**: All ranks must participate in collective operations in the
   same order.

1. **Tensor Types**: HCCL supports common PyTorch types:

    - `int8`, `int16`, `int32`, `int64`
    - `uint8`, `uint16`, `uint32`, `uint64`
    - `float16`, `float32`, `float64`
    - `bfloat16`

1. **Reduce Operations**: Supported reduce operations are `SUM`, `PRODUCT`, `MAX`, and
   `MIN`.

### YR Direct Transport

1. **Environment Setup**: Always set `YR_DS_WORKER_HOST` and `YR_DS_WORKER_PORT` before
   using YR transport.

1. **Single Device Type**: All tensors in one RDT object must have the same device type
   (all CPU or all NPU).

1. **Decorator Usage**: Use `@ray.method(tensor_transport="YR")` on actor methods that
   return or receive tensors to be transported.

1. **Health Check**: Use `actor_has_tensor_transport()` to verify an actor has YR
   transport properly configured before sending tensors.

1. **Zero-Copy for CPU**: CPU tensors use a packed binary format for efficient zero-copy
   transfer via shared memory.

## Troubleshooting

### HCCL Issues

**Problem**: "Unable to meet other processes at the rendezvous store"

**Solution**:

- Ensure all ranks are calling `create_collective_group()` with the same group name
- Verify all actors have the correct resources assigned (`resources={"NPU": 1}`)
- Check that the same number of actors are specified in `create_collective_group()`

**Problem**: "Collective ops must use the same device as communicator initialization"

**Solution**: Ensure the tensor you're passing is on the same NPU device that was
current when the `HCCLGroup` was created.

### YR Transport Issues

**Problem**: "YuanRong datasystem worker env not set"

**Solution**: Set the required environment variables:

```bash
export YR_DS_WORKER_HOST="127.0.0.1"
export YR_DS_WORKER_PORT="31502"
```

**Problem**: "Missing optional dependency 'datasystem'"

**Solution**: Install the YR extras:

```bash
pip install "ray-ascend[yr]"
```

**Problem**: "Failed to initialize YuanRong Datasystem client"

**Solution**:

- Verify the YR DataSystem worker is running at the specified host and port
- Check network connectivity
- Verify etcd is running and accessible

## FAQ

**Q: Can I use ray-ascend without NPUs?**

A: Yes! The YR transport supports CPU tensors without requiring NPU hardware or CANN.
Just install `ray-ascend[yr]` and use CPU tensors.

**Q: What Ray versions are compatible?**

A: ray-ascend should be used with the same Ray version. Check the release notes for
specific version compatibility.

**Q: Do I need to use both HCCL and YR transport together?**

A: No, they can be used independently. Use HCCL for collective operations, YR for direct
point-to-point tensor transfer, or both together as needed.

**Q: How do I choose between HCCL send/recv and YR transport?**

A: Use HCCL send/recv when you already have a collective group set up and need
fine-grained point-to-point within that group. Use YR transport for general-purpose
tensor transfer between any actors, with built-in zero-copy optimization.

## Contributing

For information on contributing to ray-ascend, see the
[Developer Guide](../developer_guide/index.md) and
[Contributing Guide](../developer_guide/contributing.md).

## Additional Resources

- [Ray Documentation](https://docs.ray.io/)
- [Ascend Documentation](https://www.hiascend.com/)
- [OpenYuanRong DataSystem](https://pages.openeuler.openatom.cn/openyuanrong-datasystem/)
- [GitHub Repository](https://github.com/Ascend/ray-ascend)
