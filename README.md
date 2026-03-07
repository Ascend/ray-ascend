<p align="center">
  <picture>
    <img alt="ray-ascend" src="./docs/logo/ray-ascend-logo.png" width=55%>
  </picture>
</p>

<h3 align="center">
Ray Ascend Plugin
</h3>

<p align="center">
| <a href="https://www.hiascend.com/en/"><b>About Ascend</b></a> | <a href="https://github.com/Ascend/ray-ascend/blob/master/README.md"><b>Documentation</b></a> |
</p>

## Overview

`ray-ascend` is a community maintained hardware plugin to support advanced [Ray](https://github.com/ray-project/ray) features on Ascend NPU accelerators.

The default ray natively supports Ascend NPU as a pre-defined resource type to bind actors and tasks (see [ray accelerator support](https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html#id1)). As an enhancement, `ray-ascend` provides Ascend-native features on ray, such as collective communication [Huawei Collective Communication Library (HCCL)](https://www.hiascend.com/document/detail/zh/canncommercial/850/commlib/hcclug/hcclug_000001.html), [Ray Direct Transport (RDT)](https://docs.ray.io/en/latest/ray-core/direct-transport.html), etc.

## Prerequisites

- Architecture: aarch64, x86
- OS kernel: linux
- Python dependencies
  - python>=3.10, <=3.11
  - CANN==8.2.rc1
  - torch==2.7.1, torch-npu==2.7.1.post1
  - ray (the same version as ray-ascend)

## Version

| Version   | Release type             | Doc |
| --------- | ------------------------ | --- |
| 0.54.0rc1 | Latest release candidate |     |


## Quick start

### Installation

```python
pip install ray-ascend[yr]
```

### HCCL collective communication among ray actors

```python
import ray
from ray.util import collective
from ray_ascend import register_hccl_collective_backend

register_hccl_collective_backend()

@ray.remote(resources={"NPU": 1})
class RayActor:
    def __init__(self):
        register_hccl_collective_backend()

collective.create_collective_group(
    actors,
    len(actors),
    list(range(0, len(actors))),
    backend="HCCL",
    group_name="my_group",
)

# each actor broadcast in a spmd manner
collective.broadcast(tensor, src_rank=0, group_name="my_group")
```

### Transport Ascend NPU tensors via [HCCS](https://www.hiascend.com/document/detail/zh/Glossary/gls/gls_0001.html#ZH-CN_TOPIC_0000002210355753__section665813471086)

```python
import ray
from ray.util.collective import create_collective_group
from ray_ascend import register_hccl_tensor_transport

register_hccl_tensor_transport()

@ray.remote(resources={"NPU": 1})
class RayActor:
    def __init__(self):
        register_hccl_tensor_transport()

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

### Transport Ascend NPU tensors via [HCCS](https://www.hiascend.com/document/detail/zh/Glossary/gls/gls_0001.html#ZH-CN_TOPIC_0000002210355753__section665813471086) and CPU tensors via RDMA

[openYuanrong-datasystem](https://pages.openeuler.openatom.cn/openyuanrong-datasystem/docs/zh-cn/latest/index.html) (`YR`) allows users to transport NPU tensors (via HCCS) and CPU tensors (via RDMA if provided) by ray objects.

```python
import ray
from ray_ascend import register_yr_tensor_transport

register_yr_tensor_transport(["npu", "cpu"])

@ray.remote(resources={"NPU": 1})
class RayActor:
    def __init__(self):
        register_yr_tensor_transport(["npu", "cpu"])

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

See [CONTRIBUTING](./CONTRIBUTING.md) for more details, which is a step-by-step guide to help you set up development environment, build and test. Please let us know if you find a bug or request a feature by [filing an issue](https://github.com/Ascend/ray-ascend/issues).

## License

Apache License 2.0. See [LICENSE](./LICENSE) file.
