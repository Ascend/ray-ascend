# YuanRong Direct Tensor Transport

> _Last updated: 03/24/2026_

YuanRong (YR) direct transport enables efficient zero-copy transfer of both CPU and NPU
tensors between Ray actors.

## Features

- **Zero-copy transfer**: Efficient memory sharing via OpenYuanRong DataSystem
- **Dual device support**: Works with both CPU and NPU tensors
- **One-sided communication**: Pull-based model for efficient transfers
- **Automatic garbage collection**: Cleans up resources when no longer needed

## Prerequisites

Before using YR transport, ensure you have:

1. Installed ray-ascend with YR support: `pip install "ray-ascend[yr]"`
1. Installed etcd (see [Installation](installation.md))
1. Set the required environment variables:

```bash
export YR_DS_WORKER_HOST="127.0.0.1"
export YR_DS_WORKER_PORT="31502"
```

## Quick Example: YR Tensor Transport

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

## Combined HCCL + YR Transport

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

## Decorator Usage

Use `@ray.method(tensor_transport="YR")` on actor methods that return or receive tensors
to be transported via YR.

## Single Device Type Constraint

All tensors in one RDT object must have the same device type (all CPU or all NPU).

## Additional Resources

- [OpenYuanRong DataSystem Documentation](https://pages.openeuler.openatom.cn/openyuanrong-datasystem/docs/zh-cn/latest/)
