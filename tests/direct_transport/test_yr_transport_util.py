from ray_ascend.direct_transport.yr_tensor_transport_util import CPUClientAdapter


class TestCPUClientAdapterPackageUnpackage:
    """Test pack_into and unpack_from methods of CPUClientAdapter."""

    def test_pack_unpack_single_item(self):
        """Test packing and unpacking a single memoryview item."""
        # Create test data
        original_data = b"Hello, World!"
        items = [memoryview(original_data)]

        # Calculate required buffer size
        packed_size = CPUClientAdapter.calc_packed_size(items)
        target_buffer = bytearray(packed_size)
        target_mv = memoryview(target_buffer)

        # Pack data
        CPUClientAdapter.pack_into(target_mv, items)

        # Unpack data
        unpacked_items = CPUClientAdapter.unpack_from(target_mv)

        # Verify
        assert len(unpacked_items) == 1
        assert bytes(unpacked_items[0]) == original_data

    def test_pack_unpack_multiple_items(self):
        """Test packing and unpacking multiple memoryview items."""
        # Create test data
        original_data_list = [
            b"First item",
            b"Second item with more data",
        ]
        items = [memoryview(data) for data in original_data_list]

        # Calculate required buffer size
        packed_size = CPUClientAdapter.calc_packed_size(items)
        target_buffer = bytearray(packed_size)
        target_mv = memoryview(target_buffer)

        # Pack data
        CPUClientAdapter.pack_into(target_mv, items)

        # Unpack data
        unpacked_items = CPUClientAdapter.unpack_from(target_mv)

        # Verify
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

        # Calculate size
        packed_size = CPUClientAdapter.calc_packed_size(items)

        # Expected size = header (4 bytes) + 3 entries (8 bytes each) + data
        expected_size = (
            CPUClientAdapter.HEADER_SIZE
            + len(items) * CPUClientAdapter.ENTRY_SIZE
            + sum(item.nbytes for item in items)
        )

        assert packed_size == expected_size
