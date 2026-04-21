# YuanRong Direct Tensor Transport

> _Last updated: 04/20/2026_

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
1. Installed dscli: `pip install openyuanrong-datasystem>=0.8.0`

## Ray Cluster Setup

```bash
# Head node
ray start --head --resources='{"NPU": 8}'

# Worker nodes
ray start --address <head_ip>:6379 --resources='{"NPU": 8}'
```

## Quick Start

```python
import torch
import ray
from ray_ascend import register_yr_tensor_transport

ray.init()

# Initialize YR backend (uses default metastore mode, no env vars needed)
register_yr_tensor_transport(["npu", "cpu"])

@ray.remote(resources={"NPU": 1})
class NPUActor:
    def __init__(self):
        register_yr_tensor_transport(["npu", "cpu"])

    @ray.method(tensor_transport="YR")
    def get_tensor(self):
        import torch
        import torch_npu
        return torch.randn(1024, 1024, device="npu")

# Create actors and transfer tensor
sender = NPUActor.remote()
receiver = NPUActor.remote()
tensor = ray.get(sender.get_tensor.remote())
print("Tensor shape:", tensor.shape)
```

## Initialization Modes

YR backend supports two initialization modes with different metadata storage backends,
controlled by `YR_DS_INIT_MODE` environment variable. For details, see
[OpenYuanrong DataSystem Cluster Management](https://pages.openeuler.openatom.cn/openyuanrong-datasystem/docs/zh-cn/latest/design_document/cluster_management.html).

| Mode        | Description                                 | Use Case                     |
| ----------- | ------------------------------------------- | ---------------------------- |
| `metastore` | Built-in metastore service via YR DS worker | Default, no external infra   |
| `etcd`      | External etcd for metadata storage          | Existing etcd infrastructure |

### Environment Variables

Set these in driver node only. YR backend coordinator will propagate configuration to
all worker nodes via Ray placement group.

| Variable               | Default     | Description                                 | Mode      |
| ---------------------- | ----------- | ------------------------------------------- | --------- |
| `YR_DS_INIT_MODE`      | `metastore` | Initialization mode (`metastore` or `etcd`) | Both      |
| `YR_DS_WORKER_PORT`    | `31501`     | YR DS worker port                           | Both      |
| `YR_DS_METASTORE_PORT` | `2379`      | Metastore service port                      | metastore |
| `YR_DS_ETCD_ADDRESS`   | -           | Etcd address (e.g., `10.0.0.1:2379`)        | etcd      |
| `YR_DS_WORKER_ARGS`    | -           | Additional dscli arguments                  | Both      |

**Worker Args Options for NPU:**

- `--shared_memory_size_mb`: Shared memory size in MB for tensor storage
- `--remote_h2d_device_ids`: Enable RH2D for cross-node transfer (comma-separated NPU
  IDs)
- `--enable_huge_tlb`: Enable huge page memory (required for >21GB on 910B)

For more YR DS worker arguments, see
[dscli documentation](https://pages.openeuler.openatom.cn/openyuanrong-datasystem/docs/zh-cn/latest/deployment/dscli.html).

**Set in Python (before `ray.init()`):**

```python
import os
os.environ["YR_DS_WORKER_ARGS"] = "--shared_memory_size_mb 16384 --remote_h2d_device_ids 0,1,2,3 --enable_huge_tlb true"
import ray
ray.init()  # env vars must be set before ray.init()
```

**Set in Bash:**

```bash
export YR_DS_WORKER_ARGS="--shared_memory_size_mb 16384 --remote_h2d_device_ids 0,1,2,3 --enable_huge_tlb true"
python your_script.py
```

### Metastore Mode Example

Metastore mode is the default and recommended for most use cases. No external etcd
service required - a YR DS worker automatically starts the metastore service.

```python
import os
import ray
from ray_ascend import register_yr_tensor_transport

os.environ["YR_DS_INIT_MODE"] = "metastore"
os.environ["YR_DS_WORKER_PORT"] = 31501
os.environ["YR_DS_WORKER_ARGS"] = "--shared_memory_size_mb 16384 --remote_h2d_device_ids 0,1,2,3"

ray.init()
register_yr_tensor_transport(["npu", "cpu"])
```

### Etcd Mode Example

Etcd mode requires an external etcd service. Ensure etcd is running and accessible
before starting YR backend. For etcd installation, see [Installation](installation.md).

```python
import os
import ray
from ray_ascend import register_yr_tensor_transport

os.environ["YR_DS_INIT_MODE"] = "etcd"
os.environ["YR_DS_WORKER_PORT"] = 31501
os.environ["YR_DS_ETCD_ADDRESS"] = "10.0.0.1:2379"
os.environ["YR_DS_WORKER_ARGS"] = "--shared_memory_size_mb 16384 --remote_h2d_device_ids 0,1,2,3"

ray.init()
register_yr_tensor_transport(["npu", "cpu"])
```

## Detailed Example

This example demonstrates full YR transport usage with NPU and CPU actors, including
tensor transfer via HCCS (NPU) and RDMA (CPU if available).

```python
import os
import ray
from ray_ascend import register_yr_tensor_transport

ray.init()

register_yr_tensor_transport(["npu", "cpu"])

@ray.remote(resources={"NPU": 1})
class NPUActor:
    def __init__(self):
        register_yr_tensor_transport(["npu", "cpu"])

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
    def __init__(self):
        register_yr_tensor_transport(["cpu"])

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

For advanced use cases combining HCCL collective communication with YR direct transport:

```python
import os
import ray
from ray.util.collective import create_collective_group
from ray_ascend import register_yr_tensor_transport
from ray_ascend.collective import HCCLGroup

ray.init(address="auto")

# Register both backends
ray.register_collective_backend("HCCL", HCCLGroup)
register_yr_tensor_transport(["npu", "cpu"])

@ray.remote(resources={"NPU": 1})
class RayActor:
    def __init__(self):
        import torch
        import torch_npu
        register_yr_tensor_transport(["npu", "cpu"])

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

## Cleanup

Use `cleanup_yr_resources()` for explicit cleanup, or `ray stop` will automatically
destroy YR backend:

```python
from ray_ascend.utils import cleanup_yr_resources
cleanup_yr_resources()
```

## Additional Resources

- [OpenYuanrong DataSystem Documentation](https://pages.openeuler.openatom.cn/openyuanrong-datasystem/docs/zh-cn/latest/)
