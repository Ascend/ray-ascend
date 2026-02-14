import os
import pickle
import uuid
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import ray
from ray.experimental import (
    CommunicatorMetadata,
    TensorTransportManager,
    TensorTransportMetadata,
)

if TYPE_CHECKING:
    import torch

try:
    import torch
    import torch_npu
    from yr.datasystem import DsTensorClient
except ImportError as e:
    raise ImportError(
        "The 'yr_tensor_transport' feature requires yr optional dependencies. "
        "Please install them with: pip install 'ray-ascend[yr]'"
    ) from e


@dataclass
class YRCommunicatorMetadata(CommunicatorMetadata):
    """Metadata for the YR communicator."""


@dataclass
class YRTransportMetadata(TensorTransportMetadata):
    """Metadata for tensors stored in the GPU object store for YR transport.
    Args:
        ds_serialized_keys: Serialized tensor keys for YR transport.
    """

    ds_serialized_keys: bytes

    __eq__ = object.__eq__
    __hash__ = object.__hash__


class YRTensorTransport(TensorTransportManager):
    def __init__(self):
        """
        Prepares the env for lazily initializing the YR DS client.
        """
        self._ds_client = None
        self._ds_worker_host = os.getenv("YR_DS_WORKER_HOST")
        self._ds_worker_port = int(os.getenv("YR_DS_WORKER_PORT"))
        npu_ids = os.getenv("ASCEND_RT_VISIBLE_DEVICES")
        if len(npu_ids) > 1:
            warnings.warn(
                f"Data system requires exactly 1 NPU, but detected {len(npu_ids)} "
                f"NPUs. Will use the first NPU (ID: {npu_ids[0]}) to connect "
                "to the data system"
            )

    def tensor_transport_backend(self) -> str:
        return "YR"

    @staticmethod
    def is_one_sided() -> bool:
        return True

    @staticmethod
    def can_abort_transport() -> bool:
        return False

    def get_ds_client(self):
        """
        Creates a YR DS client if it does not already exist.
        """
        if self._ds_client is not None:
            return self._ds_client

        try:
            self._ds_client = DsTensorClient(
                host=self._ds_worker_host,
                port=self._ds_worker_port,
                device_id=0,
                connect_timeout_ms=60000,
            )
            self._ds_client.init()
        except Exception as e:
            self._ds_client = None
            raise RuntimeError(
                f"Failed to initialize YR DS client at"
                f"{self._ds_worker_host}:{self._ds_worker_port}. "
                f"Error: {e}"
            ) from e

        return self._ds_client

    def actor_has_tensor_transport(self, actor: "ray.actor.ActorHandle") -> bool:
        # TODO(haichuan): Check if yr ds worker is connectable.
        return True

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
        ds_client = self.get_ds_client()
        try:
            ds_client.dev_mset(keys=keys, tensors=tensors)
        except Exception as e:
            raise RuntimeError(
                f"Failed to dev_mset {len(tensors)} tensors to "
                f"{self._ds_worker_host}:{self._ds_worker_port}. Error: {e}"
            ) from e

        return pickle.dumps(keys)

    def extract_tensor_transport_metadata(
        self,
        obj_id: str,
        gpu_object: List["torch.Tensor"],
    ) -> YRTransportMetadata:

        device = None
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
            tensor_device=device,
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
    ) -> List["torch.Tensor"]:
        # create empty tensors from meta data
        tensors = []
        device = tensor_transport_metadata.tensor_device
        for meta_data in tensor_transport_metadata.tensor_meta:
            shape, dtype = meta_data
            import torch

            tensor = torch.empty(size=shape, dtype=dtype, device=device)
            tensors.append(tensor)

        assert isinstance(tensor_transport_metadata, YRTransportMetadata)
        assert isinstance(communicator_metadata, YRCommunicatorMetadata)

        serialized_keys = tensor_transport_metadata.ds_serialized_keys
        keys = pickle.loads(serialized_keys)

        ds_client = self.get_ds_client()
        try:
            ds_client.dev_mget(
                keys=keys,
                tensors=tensors,
                sub_timeout_ms=30000,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to dev_mget {len(tensors)} tensors from "
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
            "DS transport does not support send_multiple_tensors,"
            "since it is a one-sided transport."
        )

    def garbage_collect(
        self,
        obj_id: str,
        tensor_transport_meta: TensorTransportMetadata,
    ):
        assert isinstance(tensor_transport_meta, YRTransportMetadata)
        serialized_keys = tensor_transport_meta.ds_serialized_keys
        if serialized_keys is not None:
            keys = pickle.loads(serialized_keys)
            ds_client = self.get_ds_client()
            try:
                ds_client.dev_delete(keys=keys)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to dev_delete {len(keys)} keys for object {obj_id} "
                    f"at {self._ds_worker_host}:{self._ds_worker_port}. Error: {e}"
                ) from e

    def abort_transport(
        self,
        obj_id: str,
        communicator_metadata: CommunicatorMetadata,
    ):
        raise NotImplementedError("YR transport does not support aborting.")
