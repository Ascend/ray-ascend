"""
Unit tests for YR Tensor Transport Utility Classes.
Tests CPUClientAdapter and NPUClientAdapter independently from
the higher-level transport logic.
"""

from unittest.mock import MagicMock, patch

import pytest

from ray_ascend.direct_transport.yr_tensor_transport_util import (
    CPUClientAdapter,
    NPUClientAdapter,
)


class TestCPUClientAdapterPackageUnpackage:
    """Test pack_into and unpack_from methods of CPUClientAdapter."""

    def test_pack_unpack_single_item(self):
        """Test packing and unpacking a single memoryview item."""
        original_data = b"Hello, World!"
        items = [memoryview(original_data)]

        packed_size = CPUClientAdapter.calc_packed_size(items)
        target_buffer = bytearray(packed_size)
        target_mv = memoryview(target_buffer)

        CPUClientAdapter.pack_into(target_mv, items)
        unpacked_items = CPUClientAdapter.unpack_from(target_mv)

        assert len(unpacked_items) == 1
        assert bytes(unpacked_items[0]) == original_data

    def test_pack_unpack_multiple_items(self):
        """Test packing and unpacking multiple memoryview items."""
        original_data_list = [
            b"First item",
            b"Second item with more data",
        ]
        items = [memoryview(data) for data in original_data_list]

        packed_size = CPUClientAdapter.calc_packed_size(items)
        target_buffer = bytearray(packed_size)
        target_mv = memoryview(target_buffer)

        CPUClientAdapter.pack_into(target_mv, items)
        unpacked_items = CPUClientAdapter.unpack_from(target_mv)

        assert len(unpacked_items) == len(original_data_list)
        for original, unpacked in zip(original_data_list, unpacked_items):
            assert bytes(unpacked) == original

    def test_calc_packed_size(self):
        """Test that calculated packed size matches actual buffer requirements."""
        items = [
            memoryview(b"Item 1"),
            memoryview(b"Item 2 longer"),
            memoryview(b"Item 3"),
        ]

        packed_size = CPUClientAdapter.calc_packed_size(items)

        expected_size = (
            CPUClientAdapter.HEADER_SIZE
            + len(items) * CPUClientAdapter.ENTRY_SIZE
            + sum(item.nbytes for item in items)
        )

        assert packed_size == expected_size

    def test_calc_packed_size_empty_list(self):
        """Test calc_packed_size with empty list."""
        assert CPUClientAdapter.calc_packed_size([]) == CPUClientAdapter.HEADER_SIZE


class TestNPUClientAdapter:
    """Test NPUClientAdapter functionality."""

    @pytest.fixture(autouse=True)
    def skip_without_npu(self):
        """Skip tests if NPU is not available."""
        pytest.importorskip("torch_npu", reason="torch_npu is not installed")

    @pytest.fixture
    def mock_ds_client(self):
        """Mock DsTensorClient for testing."""
        client = MagicMock()
        client.init.return_value = None
        client.dev_mset.return_value = []
        client.dev_mget.return_value = []
        client.dev_delete.return_value = []
        return client

    def test_init_with_mock(self, mock_ds_client):
        """Test NPUClientAdapter initialization with mocked client."""
        with patch(
            "ray_ascend.direct_transport.yr_tensor_transport_util.DsTensorClient",
            return_value=mock_ds_client,
        ):
            adapter = NPUClientAdapter("127.0.0.1", 31502)
            assert adapter._client == mock_ds_client

    def test_delete_with_failed_keys(self, mock_ds_client):
        """Test delete operation handles failed keys."""
        mock_ds_client.dev_delete.return_value = ["failed_key"]

        with patch(
            "ray_ascend.direct_transport.yr_tensor_transport_util.DsTensorClient",
            return_value=mock_ds_client,
        ):
            adapter = NPUClientAdapter("127.0.0.1", 31502)

            with pytest.raises(RuntimeError, match="Failed to delete"):
                adapter.delete(["key1", "key2"])


class TestAdapterInterfaceConsistency:
    """Test that both adapters implement the same interface."""

    def test_adapter_has_required_methods(self):
        """Verify both adapters have the required interface methods."""
        from ray_ascend.direct_transport.yr_tensor_transport_util import (
            BaseDSAdapter,
        )

        required_methods = ["init", "put", "get", "delete"]

        # Test that BaseDSAdapter (ABC) defines required methods
        for method in required_methods:
            assert hasattr(BaseDSAdapter, method)

        # Check that CPUClientAdapter implements required methods
        with patch("ray_ascend.direct_transport.yr_tensor_transport_util.KVClient"):
            adapter = CPUClientAdapter("127.0.0.1", 31502)
            for method in required_methods:
                assert hasattr(adapter, method)
                assert callable(getattr(adapter, method))
