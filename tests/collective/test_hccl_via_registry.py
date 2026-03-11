import pytest
import ray
import torch
from ray.experimental.collective import create_collective_group
from ray.util.collective import (
    allgather,
    allreduce,
    broadcast,
    recv,
    reduce,
    reducescatter,
    send,
)
from ray.util.collective.types import ReduceOp

from ray_ascend import register_hccl_collective_backend


@pytest.fixture(scope="session")
def actors(ray_cluster_with_npu):
    world_size = 2
    group_name = "hccl_group"
    actors = [HCCLRegistryTestActor.remote(group_name) for _ in range(world_size)]

    # Create collective group using Ray's experimental interface
    create_collective_group(
        actors=actors,
        backend="HCCL",
        name=group_name,
    )

    yield actors

    for actor in actors:
        try:
            ray.get(actor.destroy.remote())
        except Exception:
            # Best-effort cleanup; rely on Ray shutdown for process teardown.
            pass


@ray.remote(resources={"NPU": 1})
class HCCLRegistryTestActor:
    def __init__(self, group_name):
        register_hccl_collective_backend()
        self.group_name = group_name

    def destroy(self):
        pass

    def get_rank(self):
        return ray.get_gpu_ids()[0] if ray.get_gpu_ids() else None

    def test_allreduce(self, tensor_data):
        """Test allreduce operation."""
        device = torch.npu.current_device()
        tensor = torch.tensor(tensor_data, dtype=torch.float32, device=device)
        allreduce(tensor, group_name=self.group_name, op=ReduceOp.SUM)
        return tensor.cpu().tolist()

    def test_broadcast(self, tensor_data, src_rank=0):
        """Test broadcast operation."""
        device = torch.npu.current_device()
        tensor = torch.tensor(tensor_data, dtype=torch.float32, device=device)
        broadcast(tensor, src_rank=src_rank, group_name=self.group_name)
        return tensor.cpu().tolist()

    def test_send(self, tensor_data, dst_rank):
        """Test send operation."""
        device = torch.npu.current_device()
        tensor = torch.tensor(tensor_data, dtype=torch.float32, device=device)
        send(tensor, dst_rank=dst_rank, group_name=self.group_name)

    def test_recv(self, tensor_shape, src_rank):
        """Test recv operation."""
        device = torch.npu.current_device()
        tensor = torch.zeros(tensor_shape, dtype=torch.float32, device=device)
        recv(tensor, src_rank=src_rank, group_name=self.group_name)
        return tensor.cpu().tolist()

    def test_allgather(self, tensor_data):
        """Test allgather operation."""
        device = torch.npu.current_device()
        tensor = torch.tensor(tensor_data, dtype=torch.float32, device=device)
        # Create output tensor list for each rank
        tensor_list = [torch.zeros_like(tensor) for _ in range(2)]
        allgather(tensor_list=tensor_list, tensor=tensor, group_name=self.group_name)
        return [t.cpu().tolist() for t in tensor_list]

    def test_reduce(self, tensor_data, dst_rank=0):
        """Test reduce operation."""
        device = torch.npu.current_device()
        tensor = torch.tensor(tensor_data, dtype=torch.float32, device=device)
        reduce(tensor, dst_rank=dst_rank, group_name=self.group_name, op=ReduceOp.SUM)
        return tensor.cpu().tolist()

    def test_reducescatter(self, tensor_data_list):
        """Test reducescatter operation."""
        device = torch.npu.current_device()
        # tensor_data_list is a list of tensors (one per rank)
        tensor_list = [
            torch.tensor(data, dtype=torch.float32, device=device)
            for data in tensor_data_list
        ]
        output_tensor = torch.zeros_like(tensor_list[0])
        reducescatter(
            tensor=output_tensor,
            tensor_list=tensor_list,
            group_name=self.group_name,
            op=ReduceOp.SUM,
        )
        return output_tensor.cpu().tolist()


def test_allreduce(actors):
    """Test allreduce collective communication."""
    world_size = 2
    assert len(actors) == world_size, f"Expected {world_size} actors, got {len(actors)}"

    rank0_data = [1.0, 2.0, 3.0]
    rank1_data = [4.0, 5.0, 6.0]
    results = ray.get(
        [
            actors[0].test_allreduce.remote(rank0_data),
            actors[1].test_allreduce.remote(rank1_data),
        ]
    )
    expected = [5.0, 7.0, 9.0]
    for result in results:
        assert result == expected, f"Allreduce failed: {result} != {expected}"


def test_broadcast(actors):
    """Test broadcast collective communication."""
    world_size = 2
    assert len(actors) == world_size, f"Expected {world_size} actors, got {len(actors)}"

    root_tensor = [10.0, 20.0]
    results = ray.get(
        [actor.test_broadcast.remote(root_tensor, src_rank=0) for actor in actors]
    )
    for result in results:
        assert result == root_tensor, f"Broadcast failed: {result} != {root_tensor}"


def test_allgather(actors):
    """Test allgather collective communication."""
    world_size = 2
    assert len(actors) == world_size, f"Expected {world_size} actors, got {len(actors)}"

    rank0_data = [1.0, 2.0]
    rank1_data = [3.0, 4.0]
    results = ray.get(
        [
            actors[0].test_allgather.remote(rank0_data),
            actors[1].test_allgather.remote(rank1_data),
        ]
    )
    for i, result in enumerate(results):
        result_flattened = [item for sublist in result for item in sublist]
        all_values = sorted(result_flattened)
        expected_values = sorted([1.0, 2.0, 3.0, 4.0])
        assert (
            all_values == expected_values
        ), f"Allgather failed for rank {i}: {all_values} != {expected_values}"
        assert (
            len(result) == 2
        ), f"Allgather failed for rank {i}: expected 2 gathered tensors, got {len(result)}"


def test_reduce(actors):
    """Test reduce collective communication."""
    world_size = 2
    assert len(actors) == world_size, f"Expected {world_size} actors, got {len(actors)}"

    rank0_data = [1.0, 2.0, 3.0]
    rank1_data = [4.0, 5.0, 6.0]
    results = ray.get(
        [
            actors[0].test_reduce.remote(rank0_data, dst_rank=0),
            actors[1].test_reduce.remote(rank1_data, dst_rank=0),
        ]
    )
    expected_root = [5.0, 7.0, 9.0]
    assert (
        results[0] == expected_root
    ), f"Reduce failed for root rank: {results[0]} != {expected_root}"


def test_reducescatter(actors):
    """Test reducescatter collective communication."""
    world_size = 2
    assert len(actors) == world_size, f"Expected {world_size} actors, got {len(actors)}"

    rank0_data = [1.0, 2.0, 3.0]
    rank1_data = [4.0, 5.0, 6.0]
    results = ray.get(
        [
            actors[0].test_reducescatter.remote([rank0_data, rank1_data]),
            actors[1].test_reducescatter.remote([rank0_data, rank1_data]),
        ]
    )
    expected_rank0 = [2.0, 4.0, 6.0]
    expected_rank1 = [8.0, 10.0, 12.0]
    assert (
        results[0] == expected_rank0
    ), f"Reducescatter failed for rank 0: {results[0]} != {expected_rank0}"
    assert (
        results[1] == expected_rank1
    ), f"Reducescatter failed for rank 1: {results[1]} != {expected_rank1}"


def test_send_recv(actors):
    """Test send/recv point-to-point communication."""
    world_size = 2
    assert len(actors) == world_size, f"Expected {world_size} actors, got {len(actors)}"

    tensor_data = [7.0, 8.0, 9.0]
    tensor_shape = (3,)

    send_task = actors[0].test_send.remote(tensor_data, dst_rank=1)
    recv_task = actors[1].test_recv.remote(tensor_shape, src_rank=0)

    ray.get(send_task)
    result = ray.get(recv_task)

    assert result == tensor_data, f"Send/recv failed: {result} != {tensor_data}"
