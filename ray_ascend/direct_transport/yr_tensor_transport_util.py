import struct
import warnings
from concurrent.futures import ThreadPoolExecutor

import torch

try:
    from yr.datasystem import KVClient

    YR_AVAILABLE = True
except ImportError:
    KVClient = None
    YR_AVAILABLE = False
    warnings.warn(
        "The 'yr_tensor_transport' feature requires optional dependencies"
        "'datasystem', Install with: 'pip install openyuanrong-datasystem'",
        RuntimeWarning,
    )

try:
    import torch_npu  # noqa: F401
    from yr.datasystem import DsTensorClient

    NPU_AVAILABLE = True
except ImportError:
    DsTensorClient = None
    NPU_AVAILABLE = False
    warnings.warn(
        "The 'yr_tensor_transport' feature requires optional dependencies "
        "'torch_npu'. CPU-only paths can still work, but NPU transport "
        "will be unavailable. Install with: 'pip install torch-npu'",
        RuntimeWarning,
    )


from abc import ABC, abstractmethod

from ray_ascend.utils.serial_utils import _decoder, _encoder


def raise_if_failed(failed_keys, action):
    if failed_keys:
        raise RuntimeError(f"Failed to {action} keys: {failed_keys}")


class BaseDSAdapter(ABC):
    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def put(self, keys, tensors):
        pass

    @abstractmethod
    def get(self, keys, tensors):
        pass

    @abstractmethod
    def delete(self, keys):
        pass


class CPUClientAdapter(BaseDSAdapter):
    # Header: number of entries (uint32, little-endian)
    HEADER_FMT = "<I"
    HEADER_SIZE = struct.calcsize(HEADER_FMT)
    # Entry: (payload_offset: uint32, payload_size: uint32)
    ENTRY_FMT = "<II"
    ENTRY_SIZE = struct.calcsize(ENTRY_FMT)

    DS_MAX_WORKERS = 2

    def __init__(self, host, port):
        if not YR_AVAILABLE:
            raise RuntimeError(
                "Missing optional dependency 'datasystem'. Install with: "
                "'pip install openyuanrong-datasystem' to use CPUClientAdapter."
            )
        self._client = KVClient(host=host, port=port)
        self.local_tensors = []

    def init(self):
        self._client.init()

    @classmethod
    def calc_packed_size(cls, items: list[memoryview]) -> int:
        """
        Calculate the total size (in bytes) required to pack a list of memoryview items
        into the structured binary format used by pack_into.

        Args:
            items: List of memoryview objects to be packed.

        Returns:
            Total buffer size in bytes.
        """
        return (
            cls.HEADER_SIZE
            + len(items) * cls.ENTRY_SIZE
            + sum(item.nbytes for item in items)
        )

    @classmethod
    def pack_into(cls, target: memoryview, items: list[memoryview]):
        """
        Pack multiple contiguous buffers into a single buffer.
            ┌───────────────┐
            │ item_count    │  uint32
            ├───────────────┤
            │ entries       │  N * item entries
            ├───────────────┤
            │ payload blob  │  N * concatenated buffers
            └───────────────┘

        Args:
            target (memoryview): A writable memoryview returned by StateValueBuffer.MutableData().
                It must be large enough to accommodate the total number of bytes of HEADER + ENTRY_TABLE + all items.
                This buffer is usually mapped to shared memory or Zero-Copy memory area.
            items (List[memoryview]): List of read-only memory views (e.g., from serialized objects).
                Each item must support the buffer protocol and be readable as raw bytes.

        """
        struct.pack_into(cls.HEADER_FMT, target, 0, len(items))

        entry_offset = cls.HEADER_SIZE
        payload_offset = cls.HEADER_SIZE + len(items) * cls.ENTRY_SIZE

        target_tensor = torch.frombuffer(target, dtype=torch.uint8)

        for item in items:
            struct.pack_into(
                cls.ENTRY_FMT, target, entry_offset, payload_offset, item.nbytes
            )
            src_tensor = torch.frombuffer(item, dtype=torch.uint8)
            target_tensor[payload_offset : payload_offset + item.nbytes].copy_(
                src_tensor
            )
            entry_offset += cls.ENTRY_SIZE
            payload_offset += item.nbytes

    @classmethod
    def unpack_from(cls, source: memoryview) -> list[memoryview]:
        """
        Unpack multiple contiguous buffers from a single packed buffer.
        Args:
            source (memoryview): The packed source buffer.
        Returns:
            list[memoryview]: List of unpacked contiguous buffers.
        """
        mv = memoryview(source)
        item_count = struct.unpack_from(cls.HEADER_FMT, mv, 0)[0]
        offsets = []
        for i in range(item_count):
            offset, length = struct.unpack_from(
                cls.ENTRY_FMT, mv, cls.HEADER_SIZE + i * cls.ENTRY_SIZE
            )
            offsets.append((offset, length))
        return [mv[offset : offset + length] for offset, length in offsets]

    def put(self, keys: list[str], tensors: list[torch.Tensor]):
        """Store multiple objects in zero-copy mode using parallel serialization and buffer packing.

        Args:
            keys (list[str]): List of string keys under which the objects will be stored.
            tensors (list[torch.Tensor]): List of tensors to store.
        """
        items_list = [[memoryview(b) for b in _encoder.encode(obj)] for obj in tensors]
        packed_sizes = [self.calc_packed_size(items) for items in items_list]
        buffers = self._client.mcreate(keys, packed_sizes)
        tasks = [
            (target.MutableData(), item)
            for target, item in zip(buffers, items_list, strict=True)
        ]
        with ThreadPoolExecutor(max_workers=self.DS_MAX_WORKERS) as executor:
            list(executor.map(lambda p: self.pack_into(*p), tasks))
        self._client.mset_buffer(buffers)

    def get(self, keys: list[str], tensors: list[torch.Tensor]):
        """Retrieve multiple objects in zero-copy mode by directly deserializing from shared memory buffers.

        Args:
            keys (list[str]): List of string keys to retrieve from storage.
            tensors (list[torch.Tensor]): Pre-allocated list of tensors to hold the retrieved data. The length of this list should match the number of keys.

        """
        buffers = self._client.get_buffers(keys)
        for i, buffer in enumerate(buffers):
            if buffer is None:
                raise RuntimeError(f"Failed to get key: {keys[i]}")
            tensors[i] = _decoder.decode(self.unpack_from(buffer))

    def delete(self, keys):
        failed_keys = self._client.delete(keys=keys)
        raise_if_failed(failed_keys, "delete")

    def health_check(self):
        return self._client.health_check().is_ok()


class NPUClientAdapter(BaseDSAdapter):
    def __init__(self, host, port):
        if not NPU_AVAILABLE:
            raise RuntimeError(
                "Missing optional dependency 'datasystem' or NPU support. Install with: "
                "'pip install torch-npu' and 'pip install openyuanrong-datasystem' "
                "to ensure NPU support is available."
            )
        self._client = DsTensorClient(
            host=host,
            port=port,
            device_id=0,
            connect_timeout_ms=60000,
        )

    def init(self):
        self._client.init()

    def put(self, keys, tensors):
        failed_keys = self._client.dev_mset(keys=keys, tensors=tensors)
        raise_if_failed(failed_keys, "put")

    def get(self, keys, tensors):
        failed_keys = self._client.dev_mget(keys=keys, tensors=tensors)
        raise_if_failed(failed_keys, "get")

    def delete(self, keys):
        failed_keys = self._client.dev_delete(keys=keys)
        raise_if_failed(failed_keys, "delete")
