"""
ray-ascend is a community maintained hardware plugin to support advanced Ray features
on Ascend NPU accelerators.
"""

from typing import List

from ray_ascend import _version

__all__ = [
    "__version__",
    "__commit__",
    "register_yr_tensor_transport",
    "register_hccl_collective_backend",
    "register_hccl_tensor_transport",
    "register_hixl_tensor_transport",
]

__commit__ = _version.commit
__version__ = _version.version


def register_yr_tensor_transport(devices: List[str] = ["npu", "cpu"]) -> None:
    """
    Register YR tensor transport for Ray and initialize YR backend.

    This function should be called in the driver process to initialize the YR backend
    and register tensor transport. It must also be called in each Ray actor's __init__
    to register the transport for that actor process.

    YR backend initialization uses environment variables:
        - YR_DS_INIT_MODE: "metastore" (default) or "etcd"
        - YR_DS_WORKER_PORT: DS worker port (default: 31501)
        - YR_DS_METASTORE_PORT: Metastore port (default: 2379, metastore mode)
        - YR_DS_ETCD_ADDRESS: Etcd address (required for etcd mode)
        - YR_DS_WORKER_ARGS: Additional dscli arguments

    Args:
        devices: List of device types to support. Can be:
            - ["npu"] for NPU tensors only
            - ["npu", "cpu"] for NPU and CPU tensors
            - ["cpu"] for CPU tensors only

    Example:
        import os
        import ray
        from ray_ascend import register_yr_tensor_transport

        # Set environment variables before calling (driver only)
        os.environ["YR_DS_INIT_MODE"] = "metastore"
        os.environ["YR_DS_WORKER_PORT"] = "31501"

        # Connect to Ray cluster
        ray.init()

        # Initialize YR backend and register tensor transport (driver)
        register_yr_tensor_transport(["npu", "cpu"])

        @ray.remote(resources={"NPU": 1})
        class RayActor:
            def __init__(self):
                # Register transport in each actor process
                register_yr_tensor_transport(["npu", "cpu"])

            @ray.method(tensor_transport="YR")
            def transfer_npu_tensor_via_hccs(self):
                return torch.zeros(1024, device="npu")

            @ray.method(tensor_transport="YR")
            def transfer_cpu_tensor_via_rdma(self):
                return torch.zeros(1024)
    """
    if devices is None:
        raise ValueError(
            "devices cannot be None. Specify a list of device types, e.g., ['npu', 'cpu']"
        )

    import torch

    try:
        from ray.experimental import register_tensor_transport

        from ray_ascend.direct_transport.yr_tensor_transport import YRTensorTransport
        from ray_ascend.utils.yr_utils import ensure_yr_backend_initialized
    except ImportError as e:
        raise ImportError(
            "YR tensor transport requires the [yr] extra dependency. "
            "Please install it with: pip install ray-ascend[yr]"
        ) from e

    # Initialize YR backend (only in driver process, actors will reuse existing backend)
    # The backend initialization happens via YRBackendCoordinator which is a named actor
    # with get_if_exists=True, so it only initializes once across the cluster.
    ensure_yr_backend_initialized()

    # Register tensor transport
    register_tensor_transport("YR", devices, YRTensorTransport, torch.Tensor)


def register_hccl_collective_backend() -> None:
    """
    Register HCCL collective backend for Ray.

    Requires Ray >= 2.56. For older Ray versions, this API is not available.

    This function must be called in each Ray worker/actor process
    before using HCCL collective operations.

    Example:
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
    """
    try:
        from ray.util.collective.backend_registry import (
            _global_registry,
            register_collective_backend,
        )
    except ImportError as e:
        import ray

        raise RuntimeError(
            f"register_hccl_collective_backend requires Ray >= 2.56, "
            f"but Ray {ray.__version__} is installed. "
            f"Please upgrade: pip install 'ray>=2.56'"
        ) from e

    from .collective.hccl_collective_group import HCCLGroup

    # Idempotent: safe to call from both driver and each actor's __init__.
    if _global_registry.is_registered("HCCL"):
        return

    register_collective_backend("HCCL", HCCLGroup)


def register_hccl_tensor_transport() -> None:
    """
    Register HCCL backend and tensor transport for Ray.

    Requires Ray >= 2.56. For older Ray versions, this API is not available.

    This function must be called in each Ray worker/actor process
    before using HCCL collective operations or tensor transport.

    Example:
        from ray_ascend import register_hccl_tensor_transport

        register_hccl_tensor_transport()

        @ray.remote(resources={"NPU": 1})
        class RayActor:
            def __init__(self):
                register_hccl_tensor_transport()

            @ray.method(tensor_transport="HCCL")
            def transfer_npu_tensor_via_hccs(self):
                return torch.tensor([1, 2, 3]).npu()
    """
    register_hccl_collective_backend()

    import torch
    from ray.experimental import register_tensor_transport
    from ray.experimental.rdt.util import transport_manager_info

    from .direct_transport.hccl_tensor_transport import HCCLTensorTransport

    # Idempotent: Ray auto-syncs transport_manager_info into actors on startup.
    if "HCCL" in transport_manager_info:
        return

    register_tensor_transport("HCCL", ["npu"], HCCLTensorTransport, torch.Tensor)


def register_hixl_tensor_transport(devices: List[str] = ["npu", "cpu"]) -> None:
    """Register HIXL tensor transport for Ray RDT on Ascend NPU.

    This function should be called in the driver process to register the
    HIXL transport backend. It must also be called in each Ray actor's
    __init__ to register the transport for that actor process.

    HIXL uses RDMA READ for zero-copy tensor transfer between Ascend NPU
    nodes. Unlike YR (which uses a DataSystem intermediary), HIXL is a
    true one-sided RDMA transport where the receiver directly reads the
    sender's memory.

    Requirements:
        - HIXL wheel installed: pip install hixl
        - CANN driver and runtime installed on all NPU nodes
        - RDMA/HCCS links established between nodes

    Args:
        devices: List of device types to support. Can be:
            - ["npu"] for NPU tensors only
            - ["npu", "cpu"] for NPU and CPU tensors
            - ["cpu"] for CPU tensors only

    Example:
        import ray
        from ray_ascend import register_hixl_tensor_transport

        ray.init()
        register_hixl_tensor_transport(["npu", "cpu"])

        @ray.remote(resources={"NPU": 1})
        class RayActor:
            def __init__(self):
                register_hixl_tensor_transport(["npu", "cpu"])

            @ray.method(tensor_transport="HIXL")
            def transfer_npu_tensor_via_rdma(self):
                return torch.zeros(1024, device="npu")
    """
    if devices is None:
        raise ValueError(
            "devices cannot be None. Specify a list of device types, "
            "e.g., ['npu', 'cpu']"
        )

    import torch

    try:
        from ray.experimental import register_tensor_transport

        from ray_ascend.direct_transport.hixl_tensor_transport import (
            HixlTensorTransport,
        )
    except ImportError as e:
        raise ImportError(
            "HIXL tensor transport requires the hixl package. "
            "Please install it with: "
            "pip install hixl"
        ) from e

    # Verify hixl is importable before registration.
    try:
        import hixl
    except ImportError as e:
        raise ImportError(
            "hixl module not found. HIXL tensor transport requires "
            "the hixl wheel. Please install: "
            "pip install hixl"
        ) from e

    register_tensor_transport("HIXL", devices, HixlTensorTransport, torch.Tensor)
