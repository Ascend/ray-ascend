# Todo: re-adjust the unit tests.
import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from ray_ascend.direct_transport.yr_tensor_transport import (
    YRCommunicatorMetadata,
    YRTensorTransport,
    YRTransportMetadata,
)


@pytest.fixture(scope="module", autouse=True)
def prepare_yr_env():
    os.environ["YR_DS_WORKER_HOST"] = "127.0.0.1"
    os.environ["YR_DS_WORKER_PORT"] = "31502"


@pytest.fixture(params=["cpu", "npu"])
def device_case(request):
    """Parametrizes test cases for CPU and NPU backends and skips NPU if unavailable."""
    device = request.param

    if device == "npu":
        pytest.importorskip(
            "torch_npu",
            reason="torch_npu is not installed",
        )

    return device


@pytest.fixture
def tensors(device_case):
    """Creates test tensors on the specified device."""
    if device_case == "cpu":
        return [
            torch.randn(2, 3, device="cpu"),
            torch.randn(4, device="cpu"),
        ]
    else:
        return [
            torch.randn(2, 3, device="npu"),
            torch.randn(4, device="npu"),
        ]


@pytest.fixture
def mock_client(device_case):
    """Mock backend client methods based on device type."""
    client = MagicMock()
    client.init.return_value = None

    if device_case == "cpu":
        client.mcreate.return_value = [
            MagicMock(),
            MagicMock(),
        ]
        client.mset_buffer.return_value = None
        client.get_buffers.return_value = [
            b"fake_buffer_data_1",
            b"fake_buffer_data_2",
        ]
        client.delete.return_value = []
    else:
        client.dev_mset.return_value = None
        client.dev_mget.return_value = None
        client.dev_delete.return_value = None

    return client


@pytest.fixture
def patch_client(device_case, mock_client):
    """Patch client and encoders/decoders for CPU only."""
    if device_case == "cpu":
        with (
            patch(
                "ray_ascend.direct_transport.yr_tensor_transport_util.KVClient",
                return_value=mock_client,
            ),
            patch(
                "ray_ascend.direct_transport.yr_tensor_transport_util.YR_AVAILABLE",
                True,
            ),
        ):
            with (
                patch(
                    "ray_ascend.direct_transport.yr_tensor_transport_util._encoder"
                ) as mock_encoder,
                patch(
                    "ray_ascend.direct_transport.yr_tensor_transport_util._decoder"
                ) as mock_decoder,
                patch(
                    "ray_ascend.direct_transport.yr_tensor_transport_util.CPUClientAdapter.pack_into"
                ),
                patch(
                    "ray_ascend.direct_transport.yr_tensor_transport_util.CPUClientAdapter.unpack_from",
                    return_value=[
                        b"decoded_mock_data"
                    ],  # Mock unpack_from to return something
                ),
            ):

                mock_encoder.encode.return_value = [b"mock_meta", b"mock_raw_data"]
                mock_decoder.decode.return_value = torch.tensor([1.0, 2.0, 3.0])

                yield mock_encoder, mock_decoder, mock_client
    else:
        with (
            patch(
                "ray_ascend.direct_transport.yr_tensor_transport_util.DsTensorClient",
                return_value=mock_client,
            ),
            patch(
                "ray_ascend.direct_transport.yr_tensor_transport_util.NPU_AVAILABLE",
                True,
            ),
        ):
            yield None, None, mock_client


def test_metadata_flow(device_case, tensors, patch_client):
    """Verify metadata extraction and backend mset invocation."""
    mock_encoder, mock_decoder, mock_client = patch_client
    transport = YRTensorTransport()

    meta = transport.extract_tensor_transport_metadata(
        obj_id="obj1",
        gpu_object=tensors,
    )

    assert isinstance(meta, YRTransportMetadata)
    assert len(meta.tensor_meta) == len(tensors)
    assert meta.tensor_device.type == device_case
    assert isinstance(meta.ds_serialized_keys, (bytes, bytearray))

    if device_case == "cpu":
        mock_client.mcreate.assert_called_once()
        mock_client.mset_buffer.assert_called_once()
    else:
        mock_client.dev_mset.assert_called_once()

    mock_client.init.assert_called_once()


def test_recv_multiple_tensors(device_case, tensors, patch_client):
    """Verify tensor reconstruction and correct backend get path."""
    mock_encoder, mock_decoder, mock_client = patch_client
    transport = YRTensorTransport()

    meta = transport.extract_tensor_transport_metadata(
        obj_id="obj1",
        gpu_object=tensors,
    )

    comm_meta = YRCommunicatorMetadata()

    out = transport.recv_multiple_tensors(
        obj_id="obj1",
        tensor_transport_metadata=meta,
        communicator_metadata=comm_meta,
    )

    assert len(out) == len(tensors)

    if device_case == "cpu":
        assert mock_decoder.decode.call_count > 0
        mock_client.get_buffers.assert_called_once()
        mock_client.dev_mget.assert_not_called()
    else:
        mock_client.dev_mget.assert_called_once()
        mock_client.get_buffers.assert_not_called()


def test_garbage_collect(device_case, tensors, patch_client):
    """Verify backend cleanup is called correctly per device."""
    mock_encoder, mock_decoder, mock_client = patch_client
    transport = YRTensorTransport()

    meta = transport.extract_tensor_transport_metadata(
        obj_id="obj1",
        gpu_object=tensors,
    )

    transport.garbage_collect(
        obj_id="obj1",
        tensor_transport_meta=meta,
    )

    if device_case == "cpu":
        mock_client.delete.assert_called_once()
    else:
        mock_client.dev_delete.assert_called_once()


def test_actor_has_tensor_transport(device_case, patch_client):
    """
    Tests that actor_has_tensor_transport returns True when the remote health check succeeds.
    """
    mock_actor = MagicMock()

    mock_ray_call_chain = MagicMock()
    mock_actor.__ray_call__ = mock_ray_call_chain

    mock_ray_call_chain.options.return_value = mock_ray_call_chain
    mock_ray_call_chain.remote.return_value = "mock_object_ref"

    with patch("ray.get", return_value=True) as mock_ray_get:
        transport = YRTensorTransport()

        result = transport.actor_has_tensor_transport(mock_actor)

        # Assert the final result is True
        assert result is True

        # Assert ray.get was called exactly once with the mocked object reference
        mock_ray_get.assert_called_once_with("mock_object_ref")

        # Verify options was called exactly once with the correct concurrency_group
        mock_ray_call_chain.options.assert_called_once_with(
            concurrency_group="_ray_system"
        )

        # Verify remote was called exactly once on the object returned by options()
        mock_ray_call_chain.remote.assert_called_once()
