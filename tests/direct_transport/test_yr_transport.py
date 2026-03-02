import os
import pickle
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
    os.environ["YR_DS_WORKER_PORT"] = "31501"


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
        client.mset.return_value = None
        client.get.return_value = []
        client.delete.return_value = None
    else:
        client.dev_mset.return_value = None
        client.dev_mget.return_value = None
        client.dev_delete.return_value = None

    return client


@pytest.fixture
def patch_client(device_case, mock_client):
    """Patch the correct client constructor inside YRTensorTransport."""
    path = (
        "ray_ascend.direct_transport.yr_tensor_transport_util.KVClient"
        if device_case == "cpu"
        else "ray_ascend.direct_transport.yr_tensor_transport_util.DsTensorClient"
    )

    with patch(path, return_value=mock_client):
        yield


def test_metadata_flow(device_case, tensors, mock_client, patch_client):
    """Verify metadata extraction and backend mset invocation."""
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
        mock_client.mset.assert_called_once()
    else:
        mock_client.dev_mset.assert_called_once()

    mock_client.init.assert_called_once()


def test_recv_multiple_tensors(device_case, tensors, mock_client, patch_client):
    """Verify tensor reconstruction and correct backend get path."""
    transport = YRTensorTransport()
    mock_client.get.return_value = [pickle.dumps(tensor) for tensor in tensors]

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
        mock_client.get.assert_called_once()
        mock_client.dev_mget.assert_not_called()
    else:
        mock_client.dev_mget.assert_called_once()
        mock_client.get.assert_not_called()


def test_garbage_collect(device_case, tensors, mock_client, patch_client):
    """Verify backend cleanup is called correctly per device."""
    transport = YRTensorTransport()

    mock_client.get.return_value = [pickle.dumps(t) for t in tensors]

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


def test_actor_has_tensor_transport(device_case, mock_client, patch_client):
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
