import pickle
import warnings

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
    import torch_npu
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
    def __init__(self, host, port):
        if not YR_AVAILABLE:
            raise RuntimeError(
                "Missing optional dependency 'datasystem'. Install with: "
                "'pip install openyuanrong-datasystem' to use CPUClientAdapter."
            )
        self._client = KVClient(host=host, port=port)

    def init(self):
        self._client.init()

    def put(self, keys, tensors):
        # TODO: Do zero-copy optimization later.
        values = [pickle.dumps(t) for t in tensors]
        failed_keys = self._client.mset(keys=keys, vals=values)
        raise_if_failed(failed_keys, "put")

    def get(self, keys, tensors):
        raw_tensors = self._client.get(keys=keys)
        tensors[:] = [pickle.loads(r) for r in raw_tensors]

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
