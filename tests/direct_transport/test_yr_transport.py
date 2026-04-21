# Integration tests for YR Tensor Transport.
# Tests the main YRTensorTransport class with mock clients for both CPU and NPU,
# and Ray integration features like actor health checks.

import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from ray_ascend.direct_transport.yr_tensor_transport import (
    YRCommunicatorMetadata,
    YRTensorTransport,
    YRTransportMetadata,
)


class MockDSBuilder:
    """Builder for creating mock DS clients for testing."""

    @staticmethod
    def cpu_client():
        """Create a mock CPU DS client."""
        client = MagicMock()
        client.init.return_value = None
        client.mcreate.return_value = [MagicMock(), MagicMock()]
        client.mset_buffer.return_value = None
        client.get_buffers.return_value = [
            b"fake_buffer_data_1",
            b"fake_buffer_data_2",
        ]
        client.delete.return_value = []
        client.health_check.return_value = MagicMock(is_ok=lambda: True)
        return client

    @staticmethod
    def npu_client():
        """Create a mock NPU DS client."""
        client = MagicMock()
        client.init.return_value = None
        client.dev_mset.return_value = []
        client.dev_mget.return_value = []
        client.dev_delete.return_value = []
        return client


@pytest.fixture(scope="module", autouse=True)
def prepare_yr_env():
    """Set up environment variables for YR tests."""
    os.environ["YR_DS_WORKER_HOST"] = "127.0.0.1"
    os.environ["YR_DS_WORKER_PORT"] = "31502"


class TestCPUTransport:
    """Tests for YR Tensor Transport with CPU backend."""

    @pytest.fixture
    def transport(self):
        """Create YRTensorTransport instance."""
        return YRTensorTransport()

    @pytest.fixture
    def mock_client(self):
        """Create mock CPU client."""
        return MockDSBuilder.cpu_client()

    @pytest.fixture
    def mock_encoder_decoder(self):
        """Create mock encoder and decoder for CPU tests."""
        encoder = MagicMock()
        decoder = MagicMock()
        encoder.encode.return_value = [b"mock_meta", b"mock_raw_data"]
        decoder.decode.return_value = torch.tensor([1.0, 2.0, 3.0])
        return encoder, decoder

    @pytest.fixture
    def transport_with_mocks(self, transport, mock_client, mock_encoder_decoder):
        """Patch CPU transport with mock clients."""
        encoder, decoder = mock_encoder_decoder
        with (
            patch(
                "ray_ascend.direct_transport.yr_tensor_transport_util.KVClient",
                return_value=mock_client,
            ),
            patch(
                "ray_ascend.direct_transport.yr_tensor_transport_util.YR_AVAILABLE",
                True,
            ),
            patch(
                "ray_ascend.direct_transport.yr_tensor_transport_util._encoder",
                encoder,
            ),
            patch(
                "ray_ascend.direct_transport.yr_tensor_transport_util._decoder",
                decoder,
            ),
            patch(
                "ray_ascend.direct_transport.yr_tensor_transport_util.CPUClientAdapter.pack_into"
            ),
            patch(
                "ray_ascend.direct_transport.yr_tensor_transport_util.CPUClientAdapter.unpack_from",
                return_value=[b"decoded_mock_data"],
            ),
            patch.object(
                transport,
                "_get_worker_address",
                return_value=("127.0.0.1", 31502),
            ),
        ):
            yield transport, mock_client, decoder

    @pytest.fixture
    def cpu_tensors(self):
        """Create CPU test tensors."""
        return [
            torch.randn(2, 3, device="cpu"),
            torch.randn(4, device="cpu"),
        ]

    def test_extract_tensor_transport_metadata(self, transport_with_mocks, cpu_tensors):
        """
        Test metadata extraction for CPU tensors.
        Verifies client mcreate and mset_buffer are called correctly.
        """
        transport, mock_client, _ = transport_with_mocks

        meta = transport.extract_tensor_transport_metadata(
            obj_id="obj1",
            gpu_object=cpu_tensors,
        )

        assert isinstance(meta, YRTransportMetadata)
        assert len(meta.tensor_meta) == len(cpu_tensors)
        assert meta.tensor_device == "cpu"
        assert isinstance(meta.ds_serialized_keys, (bytes, bytearray))

        mock_client.mcreate.assert_called_once()
        mock_client.mset_buffer.assert_called_once()
        mock_client.init.assert_called_once()

    def test_recv_multiple_tensors(self, transport_with_mocks, cpu_tensors):
        """Test receiving multiple tensors on CPU."""
        transport, mock_client, decoder = transport_with_mocks

        meta = transport.extract_tensor_transport_metadata("obj1", cpu_tensors)
        comm_meta = YRCommunicatorMetadata()

        out = transport.recv_multiple_tensors(
            obj_id="obj1",
            tensor_transport_metadata=meta,
            communicator_metadata=comm_meta,
        )

        assert len(out) == len(cpu_tensors)
        assert decoder.decode.call_count > 0
        mock_client.get_buffers.assert_called_once()
        mock_client.dev_mget.assert_not_called()

    def test_garbage_collect(self, transport_with_mocks, cpu_tensors):
        """Test garbage collection for CPU tensors."""
        transport, mock_client, _ = transport_with_mocks

        meta = transport.extract_tensor_transport_metadata("obj1", cpu_tensors)
        transport.garbage_collect(obj_id="obj1", tensor_transport_meta=meta)

        mock_client.delete.assert_called_once()
        mock_client.dev_delete.assert_not_called()


class TestNPUTransport:
    """Tests for YR Tensor Transport with NPU backend."""

    @pytest.fixture(autouse=True)
    def skip_without_npu(self):
        """Skip tests if NPU is not available."""
        pytest.importorskip("torch_npu", reason="torch_npu is not installed")

    @pytest.fixture
    def transport(self):
        """Create YRTensorTransport instance."""
        return YRTensorTransport()

    @pytest.fixture
    def mock_client(self):
        """Create mock NPU client."""
        return MockDSBuilder.npu_client()

    @pytest.fixture
    def transport_with_mock(self, transport, mock_client):
        """Patch NPU transport with mock client."""
        with (
            patch(
                "ray_ascend.direct_transport.yr_tensor_transport_util.DsTensorClient",
                return_value=mock_client,
            ),
            patch(
                "ray_ascend.direct_transport.yr_tensor_transport_util.NPU_AVAILABLE",
                True,
            ),
            patch.object(
                transport,
                "_get_worker_address",
                return_value=("127.0.0.1", 31502),
            ),
        ):
            yield transport, mock_client

    @pytest.fixture
    def npu_tensors(self):
        """Create NPU test tensors."""
        return [
            torch.randn(2, 3, device="npu"),
            torch.randn(4, device="npu"),
        ]

    def test_extract_tensor_transport_metadata(self, transport_with_mock, npu_tensors):
        """
        Test metadata extraction for NPU tensors.
        Verifies client dev_mset is called correctly.
        """
        transport, mock_client = transport_with_mock

        meta = transport.extract_tensor_transport_metadata(
            obj_id="obj1",
            gpu_object=npu_tensors,
        )

        assert isinstance(meta, YRTransportMetadata)
        assert len(meta.tensor_meta) == len(npu_tensors)
        assert meta.tensor_device == "npu"
        assert isinstance(meta.ds_serialized_keys, (bytes, bytearray))

        mock_client.dev_mset.assert_called_once()
        mock_client.init.assert_called_once()

    def test_recv_multiple_tensors(self, transport_with_mock, npu_tensors):
        """Test receiving multiple tensors on NPU."""
        transport, mock_client = transport_with_mock

        meta = transport.extract_tensor_transport_metadata("obj1", npu_tensors)
        comm_meta = YRCommunicatorMetadata()

        out = transport.recv_multiple_tensors(
            obj_id="obj1",
            tensor_transport_metadata=meta,
            communicator_metadata=comm_meta,
        )

        assert len(out) == len(npu_tensors)
        mock_client.dev_mget.assert_called_once()
        mock_client.get_buffers.assert_not_called()

    def test_garbage_collect(self, transport_with_mock, npu_tensors):
        """Test garbage collection for NPU tensors."""
        transport, mock_client = transport_with_mock

        meta = transport.extract_tensor_transport_metadata("obj1", npu_tensors)
        transport.garbage_collect(obj_id="obj1", tensor_transport_meta=meta)

        mock_client.dev_delete.assert_called_once()
        mock_client.delete.assert_not_called()


class TestTransportProperties:
    """Test static properties and configuration of YRTensorTransport."""

    @pytest.fixture
    def transport(self):
        """Create YRTensorTransport instance."""
        return YRTensorTransport()

    def test_tensor_transport_backend(self, transport):
        """Test that backend name is 'YR'."""
        assert transport.tensor_transport_backend() == "YR"

    def test_is_one_sided(self, transport):
        """Test that transport is one-sided."""
        assert transport.is_one_sided() is True

    def test_can_abort_transport(self, transport):
        """Test that transport cannot be aborted."""
        assert transport.can_abort_transport() is False


class TestActorHealthCheck:
    """Test actor health check functionality for Ray integration."""

    @pytest.fixture
    def transport(self):
        """Create YRTensorTransport instance."""
        return YRTensorTransport()

    @pytest.fixture
    def mock_actor(self):
        """Create a mock actor with ray call chain."""
        mock_actor = MagicMock()
        mock_ray_call_chain = MagicMock()
        mock_actor.__ray_call__ = mock_ray_call_chain

        mock_ray_call_chain.options.return_value = mock_ray_call_chain
        mock_ray_call_chain.remote.return_value = "mock_object_ref"
        return mock_actor

    def test_actor_has_tensor_transport_success(self, transport, mock_actor):
        """
        Test that actor_has_tensor_transport returns True when health check succeeds.
        Verifies correct Ray concurrency group usage and remote call pattern.
        """
        with patch("ray.get", return_value=True) as mock_ray_get:
            result = transport.actor_has_tensor_transport(mock_actor)

            assert result is True
            mock_ray_get.assert_called_once_with("mock_object_ref")

            mock_actor.__ray_call__.options.assert_called_once_with(
                concurrency_group="_ray_system"
            )
            mock_actor.__ray_call__.remote.assert_called_once()

    def test_actor_has_tensor_transport_failure(self, transport, mock_actor):
        """Test that actor_has_tensor_transport returns False when health check fails."""
        with patch("ray.get", return_value=False) as mock_ray_get:
            result = transport.actor_has_tensor_transport(mock_actor)

            assert result is False
            mock_ray_get.assert_called_once()
