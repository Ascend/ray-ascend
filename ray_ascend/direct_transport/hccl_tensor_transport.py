import torch
from ray.experimental import (
    CommunicatorMetadata,
    TensorTransportMetadata,
)
from ray.experimental.gpu_object_manager.collective_tensor_transport import (
    CollectiveTensorTransport,
)


class HCCLTensorTransport(CollectiveTensorTransport):
    def tensor_transport_backend(self) -> str:
        return "HCCL"

    def recv_multiple_tensors(
        self,
        obj_id: str,
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
    ):
        torch.npu.set_device(0)
        return super().recv_multiple_tensors(
            obj_id, tensor_transport_metadata, communicator_metadata
        )
