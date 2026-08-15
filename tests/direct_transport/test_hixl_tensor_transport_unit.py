import pytest

pytest.importorskip("hixl", reason="unit tests require the hixl wheel")

try:
    import torch
except ImportError:  # torch is optional for some assertions below; the ones
    # that genuinely need a dtype will raise a clear error at call site.
    torch = None

from ray.experimental.rdt.tensor_transport_manager import (
    CommunicatorMetadata,
    FetchRequest,
    TensorTransportManager,
    TensorTransportMetadata,
)

from ray_ascend.direct_transport.hixl_tensor_transport import (
    HixlCommunicatorMetadata,
    HixlFetchRequest,
    HixlTensorDesc,
    HixlTensorTransport,
    HixlTransportMetadata,
)


class TestTransportProperties:
    """Verify static properties and class identity without hardware."""

    def test_tensor_transport_backend(self):
        assert HixlTensorTransport().tensor_transport_backend() == "HIXL"

    def test_is_one_sided(self):
        assert HixlTensorTransport.is_one_sided() is True

    def test_can_abort_transport(self):
        assert HixlTensorTransport.can_abort_transport() is True

    def test_inherits_tensor_transport_manager(self):
        assert issubclass(HixlTensorTransport, TensorTransportManager)

    def test_send_multiple_tensors_is_not_implemented(self):
        transport = HixlTensorTransport()
        with pytest.raises(NotImplementedError, match="one-sided"):
            transport.send_multiple_tensors(
                [],
                HixlTransportMetadata(tensor_meta=[], tensor_device=None),
                HixlCommunicatorMetadata(),
            )


class TestDataClasses:
    """Verify data class definitions, inheritance, and field layout."""

    def test_communicator_metadata_inherits(self):
        assert issubclass(HixlCommunicatorMetadata, CommunicatorMetadata)

    def test_transport_metadata_inherits(self):
        assert issubclass(HixlTransportMetadata, TensorTransportMetadata)

    def test_transport_metadata_fields(self):
        meta = HixlTransportMetadata(
            tensor_meta=[((2, 3), torch.float32)],
            tensor_device="npu",
            hixl_serialized_mem_descs=b"fake",
            hixl_engine_id="10.0.0.1:12345",
            hixl_mem_generation=0,
        )
        assert meta.hixl_serialized_mem_descs == b"fake"
        assert meta.hixl_engine_id == "10.0.0.1:12345"
        assert meta.hixl_mem_generation == 0

    def test_transport_metadata_no_duplicate_base_fields(self):
        base_fields = list(TensorTransportMetadata.__dataclass_fields__)
        child_fields = list(HixlTransportMetadata.__dataclass_fields__)
        new_fields = [f for f in child_fields if f not in base_fields]
        assert "hixl_serialized_mem_descs" in new_fields
        assert "hixl_engine_id" in new_fields
        assert "hixl_mem_generation" in new_fields

    def test_tensor_desc_fields(self):
        desc = HixlTensorDesc(
            mem_handle=42, nbytes=1024, mem_type_str="npu", metadata_count=1
        )
        assert desc.mem_handle == 42
        assert desc.nbytes == 1024
        assert desc.mem_type_str == "npu"
        assert desc.metadata_count == 1

    def test_fetch_request_inherits(self):
        assert issubclass(HixlFetchRequest, FetchRequest)

    def test_fetch_request_custom_fields(self):
        req = HixlFetchRequest(
            obj_id="test_obj",
            tensors=[],
            transfer_req=123,
            remote_engine_id="10.0.0.1:12345",
            remove_tensor_descs=True,
            transport=None,
        )
        assert req.transfer_req == 123
        assert req.remote_engine_id == "10.0.0.1:12345"
        assert req.remove_tensor_descs is True
