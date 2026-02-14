from ray.experimental.gpu_object_manager.collective_tensor_transport import (
    CollectiveTensorTransport,
)


class HCCLTensorTransport(CollectiveTensorTransport):
    def tensor_transport_backend(self) -> str:
        return "HCCL"
