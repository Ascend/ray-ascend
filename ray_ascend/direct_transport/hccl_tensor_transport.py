from typing import List, Optional

import torch
from ray.experimental.rdt.collective_tensor_transport import (
    CollectiveTensorTransport,
)
from ray.experimental.rdt.tensor_transport_manager import (
    CommunicatorMetadata,
    TensorTransportMetadata,
)


class HCCLTensorTransport(CollectiveTensorTransport):
    def tensor_transport_backend(self) -> str:
        return "HCCL"

    def recv_multiple_tensors(
        self,
        obj_id: str,
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
        target_buffers: Optional[List["torch.Tensor"]] = None,
    ) -> List["torch.Tensor"]:
        torch.npu.set_device(0)
        return super().recv_multiple_tensors(  # type: ignore[no-any-return]
            obj_id, tensor_transport_metadata, communicator_metadata, target_buffers
        )
