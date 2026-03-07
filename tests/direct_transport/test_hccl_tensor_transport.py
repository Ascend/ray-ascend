import pytest
import ray
import torch
from ray.experimental.collective import create_collective_group

from ray_ascend import register_hccl_tensor_transport

register_hccl_tensor_transport()


@pytest.fixture(scope="session")
def actors(ray_cluster_with_npu):
    """Create actors with HCCL tensor transport registered."""
    world_size = 2
    group_name = "hccl_transport_group"
    actors = [
        HCCLTensorTransportTestActor.remote(group_name) for _ in range(world_size)
    ]

    # Create collective group using Ray's experimental interface
    create_collective_group(
        actors=actors,
        backend="HCCL",
        name=group_name,
    )

    yield actors


@ray.remote(resources={"NPU": 1})
class HCCLTensorTransportTestActor:
    """Test actor for HCCL tensor transport."""

    def __init__(self, group_name):
        register_hccl_tensor_transport()
        self.group_name = group_name

    @ray.method(tensor_transport="HCCL")
    def create_tensor(self):
        """Return a random tensor on NPU."""
        return torch.tensor([1, 2, 3]).npu()

    def sum(self, tensor: torch.Tensor):
        """Return sum of tensor elements."""
        return torch.sum(tensor)


def test_hccl_tensor_transport(actors):
    """Test basic tensor transport between actors."""
    sender, receiver = actors[0], actors[1]

    # Send tensor from sender to receiver
    tensor = sender.create_tensor.remote()
    result = receiver.sum.remote(tensor)

    expected_sum = 6  # 1 + 2 + 3
    actual_sum = ray.get(result)
    assert actual_sum == expected_sum, f"Expected sum {expected_sum}, got {actual_sum}"
