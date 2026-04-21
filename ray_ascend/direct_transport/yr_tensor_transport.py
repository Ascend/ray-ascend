import logging
import pickle
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional

import ray
import torch
from ray.experimental import (
    CommunicatorMetadata,
    TensorTransportManager,
    TensorTransportMetadata,
)

from ray_ascend.direct_transport.yr_tensor_transport_util import (
    CPUClientAdapter,
    NPUClientAdapter,
)
from ray_ascend.utils.yr_utils import get_yr_backend_info

logger = logging.getLogger(__name__)


@dataclass
class YRCommunicatorMetadata(CommunicatorMetadata):
    """Metadata for the YR communicator."""


@dataclass
class YRTransportMetadata(TensorTransportMetadata):
    """Metadata for tensors stored in the GPU object store for YR transport.
    Args:
        ds_serialized_keys: Serialized tensor keys for YR transport.
    """

    ds_serialized_keys: bytes = field(default=b"")

    __eq__ = object.__eq__
    __hash__ = object.__hash__


class YRTensorTransport(TensorTransportManager):
    def __init__(self):
        """Prepares for lazy init YR DS client. Config fetched from coordinator."""
        self._ds_client: dict = {}
        self._ds_worker_host: Optional[str] = None
        self._ds_worker_port: Optional[int] = None

    def tensor_transport_backend(self) -> str:
        return "YR"

    @staticmethod
    def is_one_sided() -> bool:
        return True

    @staticmethod
    def can_abort_transport() -> bool:
        return False

    def _get_worker_address(self) -> tuple[str, int]:
        """Get worker address from coordinator.

        Returns:
            Tuple of (host, port) for this node's DS worker.

        Raises:
            RuntimeError: If worker address not found in coordinator.
        """
        my_node_ip = ray.util.get_node_ip_address()
        backend_info = get_yr_backend_info()

        node_worker_addresses = backend_info.get("node_worker_addresses", {})
        worker_addr = node_worker_addresses.get(my_node_ip)
        if not worker_addr:
            raise RuntimeError(
                f"Node {my_node_ip} not found in worker addresses. "
                f"Available nodes: {list(node_worker_addresses.keys())}"
            )

        host, port_str = worker_addr.split(":")
        return host, int(port_str)

    def get_ds_client(self, device_type: str) -> Any:
        """Creates a YR DS client if it does not already exist."""
        if self._ds_client.get(device_type) is not None:
            return self._ds_client[device_type]

        host, port = self._get_worker_address()
        self._ds_worker_host = host
        self._ds_worker_port = port

        try:
            if device_type == "npu":
                self._ds_client["npu"] = NPUClientAdapter(host, port)
            else:
                self._ds_client["cpu"] = CPUClientAdapter(host, port)
            self._ds_client[device_type].init()
            logger.info(
                f"Initialized YuanRong Datasystem client for {device_type} at {host}:{port}"
            )
        except Exception as e:
            self._ds_client.pop(device_type, None)
            raise RuntimeError(
                f"Failed to initialize YuanRong Datasystem client at {host}:{port}. Error: {e}"
            ) from e

        return self._ds_client[device_type]

    def actor_has_tensor_transport(self, actor: "ray.actor.ActorHandle") -> bool:
        def __ray_actor_has_tensor_transport__(
            self: "ray.actor.ActorHandle",
        ) -> bool:
            # Check if yr.datasystem worker is healthy
            try:
                from ray.experimental.rdt.util import get_tensor_transport_manager

                return (  # type: ignore[no-any-return]
                    get_tensor_transport_manager("YR")
                    .get_ds_client("cpu")
                    .health_check()
                )
            except Exception as e:
                logger.error(f"Raise Exception during health check: {e}")
                return False

        return ray.get(  # type: ignore[no-any-return]
            actor.__ray_call__.options(concurrency_group="_ray_system").remote(
                __ray_actor_has_tensor_transport__
            )
        )

    def get_ds_metadata(self, tensors: List["torch.Tensor"]) -> bytes:
        """Get DS metadata for a set of tensors.
        Args:
            tensors: List of tensors to get metadata for.
        Returns:
            Serialized keys for the tensors in DS.
        Raises:
            RuntimeError: If the DS client fails to call dev_mset.
        """
        keys = [f"tensor_{uuid.uuid4()}" for _ in tensors]
        device_type = tensors[0].device.type
        ds_client = self.get_ds_client(device_type)
        try:
            ds_client.put(keys=keys, tensors=tensors)
            logger.info(f"Succeed to put {len(tensors)} tensors")
        except Exception as e:
            raise RuntimeError(
                f"Failed to put {len(tensors)} tensors to "
                f"{self._ds_worker_host}:{self._ds_worker_port}. Error: {e}"
            ) from e

        return pickle.dumps(keys)

    def extract_tensor_transport_metadata(
        self,
        obj_id: str,
        gpu_object: List["torch.Tensor"],
    ) -> YRTransportMetadata:

        tensor_meta = []
        if not gpu_object:
            raise ValueError("GPU object list is empty.")
        serialized_keys = self.get_ds_metadata(gpu_object)
        # We assume all tensors in one GPU object have the same device type.
        device = gpu_object[0].device
        for t in gpu_object:
            if t.device.type != device.type:
                raise ValueError(
                    "All tensors in an RDT object must have the same device type."
                )
            tensor_meta.append((t.shape, t.dtype))

        return YRTransportMetadata(
            tensor_meta=tensor_meta,
            tensor_device=device.type,
            ds_serialized_keys=serialized_keys,
        )

    def get_communicator_metadata(
        self,
        src_actor: "ray.actor.ActorHandle",
        dst_actor: "ray.actor.ActorHandle",
        backend: Optional[str] = None,
    ) -> YRCommunicatorMetadata:
        return YRCommunicatorMetadata()

    def recv_multiple_tensors(
        self,
        obj_id: str,
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
        target_buffers: Optional[List[Any]] = None,
    ) -> List["torch.Tensor"]:
        from ray.experimental.rdt.util import create_empty_tensors_from_metadata

        # Use pre-allocated target buffers if provided, otherwise create empty tensors
        tensors = target_buffers or create_empty_tensors_from_metadata(
            tensor_transport_metadata
        )

        assert isinstance(tensor_transport_metadata, YRTransportMetadata)
        assert isinstance(communicator_metadata, YRCommunicatorMetadata)

        serialized_keys = tensor_transport_metadata.ds_serialized_keys
        keys = pickle.loads(serialized_keys)

        device_type = tensor_transport_metadata.tensor_device
        ds_client = self.get_ds_client(device_type)
        try:
            ds_client.get(keys=keys, tensors=tensors)
            logger.info(f"Succeed to get {len(tensors)} tensors")
        except Exception as e:
            raise RuntimeError(
                f"Failed to get {len(tensors)} tensors from "
                f"{self._ds_worker_host}:{self._ds_worker_port}. Error: {e}"
            ) from e

        return tensors

    def send_multiple_tensors(
        self,
        tensors: List["torch.Tensor"],
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
    ):
        raise NotImplementedError(
            "Datasystem transport does not support send_multiple_tensors,"
            "since it is a one-sided transport."
        )

    def garbage_collect(
        self,
        obj_id: str,
        tensor_transport_meta: TensorTransportMetadata,
        tensors: Optional[List[Any]] = None,
    ):
        assert isinstance(tensor_transport_meta, YRTransportMetadata)
        serialized_keys = tensor_transport_meta.ds_serialized_keys
        device_type = tensor_transport_meta.tensor_device
        if serialized_keys is not None:
            keys = pickle.loads(serialized_keys)
            ds_client = self.get_ds_client(device_type)
            try:
                ds_client.delete(keys=keys)
                logger.info("Succeed to delete all keys")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to delete {len(keys)} keys for object {obj_id} "
                    f"at {self._ds_worker_host}:{self._ds_worker_port}. Error: {e}"
                ) from e

    def abort_transport(
        self,
        obj_id: str,
        communicator_metadata: CommunicatorMetadata,
    ):
        raise NotImplementedError("YuanRong transport does not support aborting.")
