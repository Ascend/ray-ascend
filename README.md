<p align="center">
  <picture>
    <img alt="ray-ascend" src="./docs/logo/ray-ascend-logo.png" width=55%>
  </picture>
</p>

<h3 align="center">
Ray Ascend Plugin
</h3>

<p align="center">
| <a href="https://www.hiascend.com/en/"><b>About Ascend</b></a> | <a href="https://ascend.github.io/ray-ascend/"><b>Documentation</b></a> |
</p>

## Overview

`ray-ascend` is a community-maintained hardware plugin that supports advanced
[Ray](https://github.com/ray-project/ray) features on Ascend NPU accelerators.

By default, Ray natively supports Ascend NPU as a predefined resource type for binding
actors and tasks (see
[Ray Accelerator Support](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html#id1)).
As an enhancement, `ray-ascend` provides Ascend-native features on Ray, such as
collective communication via
[Huawei Collective Communication Library (HCCL)](https://www.hiascend.com/document/detail/zh/canncommercial/850/commlib/hcclug/hcclug_000001.html),
[Ray Direct Transport (RDT)](https://docs.ray.io/en/latest/ray-core/direct-transport.html),
and more.

## Prerequisites

- **Architecture**: aarch64, x86
- **OS Kernel**: Linux
- **Python Dependencies**:
  - python >= 3.10, <= 3.11
  - CANN == 8.2.rc1
  - torch == 2.7.1, torch-npu == 2.7.1.post1
  - Ray (same version as ray-ascend)

## Version

| Version   | Release Type             | Doc |
| --------- | ------------------------ | --- |
| 0.54.0rc1 | Latest Release Candidate |     |

## Quick Start

### Installation

```python
pip install "ray-ascend[yr]"
```

### HCCL Collective Communication Among Ray Actors

```python
import ray
from ray.util import collective
from ray_ascend.collective import HCCLGroup

ray.register_collective_backend("HCCL", HCCLGroup)

collective.create_collective_group(
    actors,
    len(actors),
    list(range(0, len(actors))),
    backend="HCCL",
    group_name="my_group",
)

# Each actor broadcasts in SPMD manner
collective.broadcast(tensor, src_rank=0, group_name="my_group")
```

### Transport Ascend NPU Tensors via [HCCS](https://www.hiascend.com/document/detail/zh/Glossary/gls/gls_0001.html#ZH-CN_TOPIC_0000002210355753__section665813471086)

```python
import ray
from ray.util.collective import create_collective_group
from ray.experimental import register_tensor_transport
from ray_ascend.collective import HCCLGroup
from ray_ascend.direct_transport import HCCLTensorTransport

ray.register_collective_backend("HCCL", HCCLGroup)
register_tensor_transport("HCCL", ["npu"], HCCLTensorTransport)


@ray.remote(resources={"NPU": 1})
class RayActor:
    @ray.method(tensor_transport="HCCL")
    def random_tensor(self):
        return torch.zeros(1024, device="npu")

    def sum(self, tensor: torch.Tensor):
        return torch.sum(tensor)


sender, receiver = RayActor.remote(), RayActor.remote()
group = create_collective_group([sender, receiver], backend="HCCL")

tensor = sender.random_tensor.remote()
result = receiver.sum.remote(tensor)
ray.get(result)
```

### Transport Ascend NPU Tensors via [HCCS](https://www.hiascend.com/document/detail/zh/Glossary/gls/gls_0001.html#ZH-CN_TOPIC_0000002210355753__section665813471086) and CPU Tensors via RDMA

[OpenYuanRong DataSystem](https://pages.openeuler.openatom.cn/openyuanrong-datasystem/docs/zh-cn/latest/index.html)
(`YR`) allows users to transport NPU tensors (via HCCS) and CPU tensors (via RDMA if
provided) using Ray objects.

```python
import ray
from ray_ascend.direct_transport import YRTensorTransport
from ray.experimental import register_tensor_transport
register_tensor_transport("YR", ["npu", "cpu"], YRTensorTransport)

@ray.remote(resources={"NPU": 1})
class RayActor:
    @ray.method(tensor_transport="YR")
    def transfer_npu_tensor_via_hccs():
        return torch.zeros(1024, device="npu")

    @ray.method(tensor_transport="YR")
    def transfer_cpu_tensor_via_rdma():
        return torch.zeros(1024)

sender = RayActor.remote()
npu_tensor = ray.get(sender.transfer_npu_tensor_via_hccs())
cpu_tensor = ray.get(sender.transfer_cpu_tensor_via_rdma())
```

## Contributing

See [CONTRIBUTING](./CONTRIBUTING.md) and [developer guide](https://ascend.github.io/ray-ascend/developer_guide/) for more details—a step-by-step guide to help
you set up your development environment, build, and test. Please let us know if you find
a bug or request a feature by
[filing an issue](https://github.com/Ascend/ray-ascend/issues).

## License

Apache License 2.0. See [LICENSE](./LICENSE) file.
