"""
ray-ascend is a community maintained hardware plugin to support advanced Ray features
on Ascend NPU accelerators.
"""

from ray_ascend import _version

__all__ = [
    "__version__",
    "__commit__",
    "register_yr_tensor_transport",
    "register_hccl_collective_backend",
    "register_hccl_tensor_transport",
]

__commit__ = _version.commit
__version__ = _version.version


def register_yr_tensor_transport(devices=["npu", "cpu"]) -> None:
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
            - None (default) for both ["npu", "cpu"]

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
            def transfer_npu_tensor_via_hccs():
                return torch.zeros(1024, device="npu")

            @ray.method(tensor_transport="YR")
            def transfer_cpu_tensor_via_rdma():
                return torch.zeros(1024)
    """
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
    from ray.util.collective.backend_registry import register_collective_backend

    from .collective.hccl_collective_group import HCCLGroup

    register_collective_backend("HCCL", HCCLGroup)


def register_hccl_tensor_transport() -> None:
    """
    Register HCCL backend and tensor transport for Ray.

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
    import torch
    from ray.experimental import register_tensor_transport

    from .direct_transport.hccl_tensor_transport import HCCLTensorTransport

    register_hccl_collective_backend()
    register_tensor_transport("HCCL", ["npu"], HCCLTensorTransport, torch.Tensor)
