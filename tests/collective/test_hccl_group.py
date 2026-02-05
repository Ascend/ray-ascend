import pytest
import ray
import torch
from ray.util.collective.types import (
    AllGatherOptions,
    BroadcastOptions,
    RecvOptions,
    ReduceOp,
    ReduceOptions,
    ReduceScatterOptions,
    SendOptions,
)

from ray_ascend.collective.hccl_collective_group import HCCLGroup


@pytest.fixture(scope="session")
def ray_cluster():
    world_size = 2
    if torch.npu.device_count() < world_size:
        pytest.skip("Not enough NPU devices for HcclGroup tests")
    if not ray.is_initialized():
        try:
            ray.init(ignore_reinit_error=True, resources={"NPU": world_size})
        except ValueError:
            # Likely connecting to an existing cluster; do not pass resources.
            ray.init(ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture(scope="session")
def actors(ray_cluster):
    world_size = 2
    group_name = "hccl_group"
    actors = [
        HCCLTestActor.remote(rank, world_size, group_name) for rank in range(world_size)
    ]
    yield actors

    for actor in actors:
        try:
            ray.get(actor.destroy.remote())
        except Exception:
            # Best-effort cleanup; rely on Ray shutdown for process teardown.
            pass


@ray.remote(resources={"NPU": 1})
class HCCLTestActor:
    def __init__(self, rank, world_size, group_name="test_group"):
        self.rank = rank
        self.world_size = world_size
        self.group = HCCLGroup(world_size, rank, group_name)

    def destroy(self):
        self.group.destroy_group()

    def get_rank(self):
        return self.rank

    def test_allreduce(self, tensor_data):
        """Test allreduce operation."""
        tensor = torch.tensor(tensor_data, dtype=torch.float32).npu()
        self.group.allreduce(tensor)
        return tensor.cpu().tolist()

    def test_broadcast(self, tensor_data, root_rank=0):
        """Test broadcast operation."""
        tensor = torch.tensor(tensor_data, dtype=torch.float32).npu()
        broadcast_options = BroadcastOptions()
        broadcast_options.root_rank = root_rank
        broadcast_options.root_tensor = 0
        self.group.broadcast(tensor, broadcast_options=broadcast_options)
        return tensor.cpu().tolist()

    def test_send(self, tensor_data, dst_rank):
        """Test send operation."""
        tensor = torch.tensor(tensor_data, dtype=torch.float32).npu()
        send_options = SendOptions()
        send_options.dst_rank = dst_rank
        send_options.dst_gpu_index = 0
        self.group.send(tensor, send_options=send_options)

    def test_recv(self, tensor_shape, src_rank):
        """Test recv operation."""
        tensor = torch.zeros(tensor_shape, dtype=torch.float32).npu()
        recv_options = RecvOptions()
        recv_options.src_rank = src_rank
        recv_options.src_gpu_index = 0
        self.group.recv(tensor, recv_options=recv_options)
        return tensor.cpu().tolist()

    def test_allgather(self, tensor_data):
        """Test allgather operation."""
        tensor = torch.tensor(tensor_data, dtype=torch.float32).npu()
        # Create output tensors list for each rank
        output_tensors = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        allgather_options = AllGatherOptions()
        self.group.allgather(
            output_tensors, tensor, allgather_options=allgather_options
        )
        return [t.cpu().tolist() for t in output_tensors]

    def test_reduce(self, tensor_data, root_rank=0):
        """Test reduce operation."""
        tensor = torch.tensor(tensor_data, dtype=torch.float32).npu()
        reduce_options = ReduceOptions()
        reduce_options.root_rank = root_rank
        reduce_options.root_tensor = 0
        reduce_options.reduceOp = ReduceOp.SUM
        self.group.reduce(tensor, reduce_options=reduce_options)
        return tensor.cpu().tolist()

    def test_reducescatter(self, tensor_data_list):
        """Test reducescatter operation."""
        # tensor_data_list is a list of tensors (one per rank)
        input_tensors = [
            torch.tensor(data, dtype=torch.float32).npu() for data in tensor_data_list
        ]
        output_tensor = torch.zeros_like(input_tensors[0])
        reducescatter_options = ReduceScatterOptions()
        reducescatter_options.reduceOp = ReduceOp.SUM
        self.group.reducescatter(
            output_tensor, input_tensors, reducescatter_options=reducescatter_options
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
        [actor.test_broadcast.remote(root_tensor, root_rank=0) for actor in actors]
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
            actors[0].test_reduce.remote(rank0_data, root_rank=0),
            actors[1].test_reduce.remote(rank1_data, root_rank=0),
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

    ranks = ray.get([actor.get_rank.remote() for actor in actors])
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
    for idx, rank in enumerate(ranks):
        expected = expected_rank0 if rank == 0 else expected_rank1
        assert (
            results[idx] == expected
        ), f"Reducescatter failed for rank {rank}: {results[idx]} != {expected}"


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
