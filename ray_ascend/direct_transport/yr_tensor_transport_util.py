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

    def _serialize_tensor_with_pickler(self, tensor: torch.Tensor):
        """使用 Pickler 实例进行序列化"""
        # 显式转换非连续 Tensor，确保 buffer_callback 只触发一次
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
            
        oob_buffers = []
        stream = io.BytesIO()
        
        # 实例化 Pickler
        # buffer_callback: 当发现支持 Buffer 协议的大块内存时触发
        pickler = pickle.Pickler(
            stream, 
            protocol=5, 
            buffer_callback=lambda b: oob_buffers.append(b.raw())
        )
        pickler.dump(tensor)
        
        skeleton = stream.getvalue()
        raw_data = oob_buffers[0] # 对于连续 Tensor，必只有一个
        
        total_size = 4 + len(skeleton) + raw_data.nbytes
        return skeleton, raw_data, total_size

    def put(self, keys: list[str], objs: list[torch.Tensor]):
        # 1. 序列化提取元数据
        serialized_results = [self._serialize_tensor_with_pickler(obj) for obj in objs]
        packed_sizes = [res[2] for res in serialized_results]

        # 2. 申请共享内存
        ds_buffers = self._ds_client.mcreate(keys, packed_sizes)

        # 3. 填充数据 (保持之前的逻辑)
        def _pack_one(target_ds_val, res):
            skeleton, raw_data, _ = res
            target_mv = target_ds_val.MutableData()
            skel_len = len(skeleton)
            
            # [4字节长度 | 骨架字节 | 原始数据]
            struct.pack_into("<I", target_mv, 0, skel_len)
            target_mv[4 : 4 + skel_len] = skeleton
            
            offset = 4 + skel_len
            # 使用 torch 快速搬运
            target_tensor = torch.frombuffer(target_mv, dtype=torch.uint8)
            src_tensor = torch.frombuffer(raw_data, dtype=torch.uint8)
            target_tensor[offset : offset + raw_data.nbytes].copy_(src_tensor)

        with ThreadPoolExecutor(max_workers=self.DS_MAX_WORKERS) as executor:
            list(executor.map(lambda p: _pack_one(*p), zip(ds_buffers, serialized_results)))

        self._ds_client.mset_buffer(ds_buffers)

    # 接收端的逻辑依然可以保持简洁
    def get(self, keys: list[str]) -> list[torch.Tensor]:
        buffers = self._ds_client.get_buffers(keys)
        results = []
        for buf in buffers:
            if buf is None:
                results.append(None)
                continue
            mv = memoryview(buf)
            skel_len = struct.unpack_from("<I", mv, 0)[0]
            skeleton = mv[4 : 4 + skel_len]
            payload = mv[4 + skel_len:]
            
            # 这里直接用 loads 即可，它底层会自动创建 Unpickler
            results.append(pickle.loads(skeleton, buffers=[payload]))
        return results

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
