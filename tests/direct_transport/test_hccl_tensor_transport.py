import pytest
import ray
import torch
from ray.experimental import register_tensor_transport
from ray.util.collective import create_collective_group

from ray_ascend.collective import HCCLGroup
from ray_ascend.direct_transport import HCCLTensorTransport


@ray.remote(resources={"NPU": 1})
class HCCLTensorActor:
    @ray.method(tensor_transport="HCCL")
    def random_tensor(self):
        return torch.randn(1000, 1000).npu()

    def sum(self, tensor: torch.Tensor):
        return torch.sum(tensor)


@pytest.fixture(scope="session")
def ray_cluster():
    world_size = 2
    if torch.npu.device_count() < world_size:
        pytest.skip("Not enough NPU devices for HCCL tensor transport test")
    if not ray.is_initialized():
        try:
            ray.init(ignore_reinit_error=True, resources={"NPU": world_size})
        except ValueError:
            ray.init(ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()


def _register_hccl_backend_and_transport():
    try:
        ray.register_collective_backend("HCCL", HCCLGroup)
    except ValueError:
        # Already registered in this process.
        pass

    try:
        register_tensor_transport("HCCL", ["npu"], HCCLTensorTransport)
    except ValueError:
        # Already registered in this process.
        pass


def test_hccl_tensor_transport_rdt(ray_cluster):
    _register_hccl_backend_and_transport()

    sender, receiver = HCCLTensorActor.remote(), HCCLTensorActor.remote()
    create_collective_group([sender, receiver], backend="HCCL")

    tensor_ref = sender.random_tensor.remote()
    result_ref = receiver.sum.remote(tensor_ref)

    result = ray.get(result_ref)
    tensor = ray.get(tensor_ref)
    expected = torch.sum(tensor)

    result = result.cpu()
    expected = expected.cpu()
    assert torch.allclose(result, expected)
