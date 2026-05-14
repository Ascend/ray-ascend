import pytest
import ray
import torch

# Default NPU resource count for HCCL tests
DEFAULT_NPU_COUNT = 2


@pytest.fixture(scope="session")
def ray_cluster_with_npu():
    """
    Initialize Ray cluster with NPU resources for HCCL tests.

    This fixture provides a Ray cluster initialized with NPU resources
    required for HCCL (Huawei Collective Communication Library) operations.
    Raises an error if torch_npu is not available or insufficient NPU devices.
    """
    if not hasattr(torch, "npu"):
        raise RuntimeError(
            "torch.npu is not available. "
            "Please install torch_npu following the guide at: "
            "https://gitcode.com/Ascend/pytorch"
        )
    if torch.npu.device_count() < DEFAULT_NPU_COUNT:
        raise RuntimeError(
            f"Not enough NPU devices for HCCL tests. "
            f"Required: {DEFAULT_NPU_COUNT}, Available: {torch.npu.device_count()}"
        )
    if not ray.is_initialized():
        try:
            ray.init(ignore_reinit_error=True, resources={"NPU": DEFAULT_NPU_COUNT})
        except ValueError:
            # Likely connecting to an existing cluster; do not pass resources.
            ray.init(ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()
