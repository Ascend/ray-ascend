# HixlTensorTransport 实现设计文档

## 1. 概述

| 项目 | 内容 |
|---|---|
| 类名 | `HixlTensorTransport` |
| 基类 | `TensorTransportManager`（Ray RDT 插件接口） |
| 参考实现 | `NixlTensorTransport`（NIXL 一侧 RDMA 传输） |
| 传输类型 | 一侧 RDMA READ（`is_one_sided=True`） |
| 设备 | Ascend NPU (`"npu"`) + CPU (`"cpu"`） |
| C++ 绑定 | `hixl_wrapper`（pybind11 模块，绑定 `hixl::Hixl`） |
| 文件路径 | `ray-ascend/ray_ascend/direct_transport/hixl_tensor_transport.py` |

---

## 2. 与 NIXL 的关键差异

以下差异决定了 HIXL 实现不能照抄 NIXL 代码：

| 环节 | NIXL 做法 | HIXL 做法 | 设计影响 |
|---|---|---|---|
| 内存描述序列化 | `nixl_agent.get_serialized_descs()` 内置方法 | **无内置序列化**，需自定义 | `HixlTransportMetadata` 用 `pickle.dumps([(data_ptr, nbytes, mem_type_str)])` |
| 远端 agent 添加 | `nixl_agent.add_remote_agent(meta)` 自动 | **显式 Connect** `hixl_wrapper.connect(remote_engine_id)` | 新增 `_connect_remote_engine` / `_disconnect_remote_engine` 方法，LRU 缓存 `_remote_engines` |
| 传输描述交换 | `nixl_agent.get_xfer_descs()` + `deserialize_descs()` 自动 | **需从 metadata 中提取远端 addr/len**，手动构造 `TransferOpDesc` | `fetch_multiple_tensors` 中反序列化 `remote_mem_descs`，构建 `[(local_addr, remote_addr, len)]` |
| 设备类型 | `"cuda"` / `"cpu"` | `"npu"` / `"cpu"` | `torch.npu.synchronize()` 替代 `torch.cuda.synchronize()` |
| 异步状态 | `check_xfer_state` → `"PROC"/"DONE"/"ERR"` | `get_transfer_status` → `"WAITING"/"COMPLETED"/"TIMEOUT"/"FAILED"` | `wait_fetch_complete` 轮询逻辑不同 |
| Agent 初始化 | `nixl_agent(actor_id, config)` 创建实例对象，同进程可有多个 agent | `hixl_wrapper.initialize(engine_id, options)` 创建**进程级全局单例**，同进程只能有一个引擎 | 惰初始化用 `_ensure_hixl_initialized()` + `_local_engine_id`，不存 agent 实例引用。NIXL 的 `self._nixl_agent` 存的是实例对象，HIXL 不需要——`hixl_wrapper` 是全局模块，所有方法都是模块级函数，直接调用即可。**此差异不影响 Python 端设计**——Ray RDT 的 `get_tensor_transport_manager("HIXL")` 本身也只创建一个 `HixlTensorTransport` Python 单例 |
| 注册句柄 | NIXL `register_memory` 返回 `reg_desc`（可序列化） | HIXL `register_mem` 返回 `mem_handle`（`int`，不可序列化） | `HixlTensorDesc` 需额外存 `nbytes` 和 `mem_type_str`，因为 `mem_handle` 不包含地址信息 |

---

## 3. 数据类设计

### 3.1 `HixlCommunicatorMetadata`

```python
@dataclass
class HixlCommunicatorMetadata(CommunicatorMetadata):
    """Metadata for the HIXL communicator."""
```

### 3.2 `HixlTransportMetadata`

```python
@dataclass
class HixlTransportMetadata(TensorTransportMetadata):
    """Metadata for tensors stored in the NPU/CPU object store for HIXL transport.

    Args:
        hixl_serialized_mem_descs: Pickle-serialized list of
            (data_ptr, nbytes, mem_type_str) tuples describing the source
            tensors' registered memory regions.
        hixl_engine_id: The local HIXL engine identifier (format: "host_ip:port")
            that the remote side uses to Connect back.
        hixl_engine_meta_version: Monotonically increasing version number bumped
            whenever memory is deregistered, so the receiver can detect stale
            descriptors.
    """

    hixl_serialized_mem_descs: Optional[bytes] = None
    hixl_engine_id: Optional[str] = None
    hixl_engine_meta_version: Optional[int] = 0

    __eq__ = object.__eq__
    __hash__ = object.__hash__
```

### 3.3 `HixlTensorDesc`

```python
@dataclass
class HixlTensorDesc:
    """Cached registration info for a single tensor storage.

    Attributes:
        mem_handle: The opaque handle returned by hixl_wrapper.register_mem.
            Represented as a Python int (uintptr_t under the hood).
        nbytes: Size of the registered memory region in bytes.
        mem_type_str: "npu" or "cpu" — used when building TransferOpDesc and
            for serialization into HixlTransportMetadata.
        metadata_count: Number of HixlTransportMetadata objects that reference
            this tensor. When it reaches zero, we call DeregisterMem.
    """

    mem_handle: Any
    nbytes: int
    mem_type_str: str
    metadata_count: int
```

### 3.4 `HixlFetchRequest`

```python
@dataclass
class HixlFetchRequest(FetchRequest):
    """HIXL-specific fetch request carrying the async transfer state.

    Returned by fetch_multiple_tensors and consumed by wait_fetch_complete.
    Resource cleanup happens in __del__ so that handles are released even if
    the caller never waits on the request.

    Args:
        obj_id: Inherited. The object ID for the transfer, used for abort checks.
        tensors: Inherited. Pre-allocated output tensors (populated before the
            transfer starts).
        transfer_req: HIXL TransferReq handle (uintptr_t → Python int).
        remote_engine_id: The remote engine ID (ip:port) that was connected
            for this transfer.
        remove_tensor_descs: Whether to remove tensor descriptors from the
            cache during cleanup (True when fetch_multiple_tensors added them).
        transport: Reference to the HixlTensorTransport instance for cleanup.
    """

    transfer_req: Any = None
    remote_engine_id: Optional[str] = None
    remove_tensor_descs: bool = False
    transport: Any = None

    def __del__(self):
        if self.transport is not None:
            self.transport._cleanup_transfer(
                self.obj_id,
                self.tensors,
                self.transfer_req,
                self.remote_engine_id,
                self.remove_tensor_descs,
            )
```

---

## 4. `HixlTensorTransport` 类完整实现

### 4.1 类声明与 `__init__`

```python
class HixlTensorTransport(TensorTransportManager):
    """HIXL Engine-based one-sided RDMA tensor transport for Ray RDT."""

    def __init__(self):
        # Lazily initialized because hixl_wrapper may not be installed on
        # nodes that are only coordinating (not participating in transfers).
        self._hixl_initialized = False
        self._local_engine_id: Optional[str] = None

        # Object IDs whose transfers have been aborted.
        self._aborted_transfer_obj_ids: set = set()
        self._aborted_transfer_obj_ids_lock = threading.Lock()

        # Mapping from tensor storage data_ptr → HixlTensorDesc.
        # Unlike _managed_meta_hixl, we only deregister tensors when ALL
        # metadata containing the tensor is freed (reference counting via
        # metadata_count).
        self._tensor_desc_cache: Dict[int, HixlTensorDesc] = {}

        # Mapping from object ID → HixlTransportMetadata.
        # Lifetime is tied to the object ref; freed when the ref goes out of
        # scope (garbage_collect is called).
        self._managed_meta_hixl: Dict[str, Any] = {}

        # Lock protecting _tensor_desc_cache and _managed_meta_hixl since they
        # can be accessed from the main task execution thread or the
        # _ray_system thread.
        self._cache_lock = threading.RLock()

        # LRU cache of remote engine IDs. When full, the least recently used
        # remote engine is evicted and Disconnect is called.
        self._remote_engines: OrderedDict = OrderedDict()

        # Incremented whenever memory is deregistered so receivers can detect
        # stale descriptors.
        self._hixl_engine_meta_version: int = 0
```

### 4.2 `tensor_transport_backend`

```python
    def tensor_transport_backend(self) -> str:
        return "HIXL"
```

### 4.3 `is_one_sided`

```python
    @staticmethod
    def is_one_sided() -> bool:
        return True  # HIXL RDMA: receiver initiates READ (one-sided)
```

### 4.4 `can_abort_transport`

```python
    @staticmethod
    def can_abort_transport() -> bool:
        return True  # TransferAsync can be interrupted via abort flag
```

### 4.5 `_ensure_hixl_initialized`

```python
    def _ensure_hixl_initialized(self):
        """Lazily initializes the HIXL engine via hixl_wrapper.

        The engine ID is constructed from the Ray actor's node IP + actor_id
        as the port component, ensuring uniqueness per actor.

        Raises:
            ImportError: If hixl_wrapper is not installed.
            RuntimeError: If HIXL initialization fails.
        """
        if self._hixl_initialized:
            return

        if hixl_wrapper is None:
            raise ImportError(
                "hixl_wrapper module not found. "
                "Please install the HIXL Engine wheel: "
                "pip install hixl_engine-0.0.1-py3-none-any.whl"
            )

        # Build a local engine ID from the Ray actor's IP address.
        # The port component uses the actor_id to ensure uniqueness.
        ctx = ray.get_runtime_context()
        actor_id = ctx.get_actor_id()
        if actor_id is None:
            # Driver process — generate a unique ID.
            import uuid
            actor_id = f"RAY-DRIVER-{uuid.uuid4()}"

        node_ip = ray.util.get_node_ip_address()
        self._local_engine_id = f"{node_ip}:{actor_id}"

        status = hixl_wrapper.initialize(self._local_engine_id, {})
        if status != hixl_wrapper.kSuccess:
            raise RuntimeError(
                f"Failed to initialize HIXL engine with id "
                f"'{self._local_engine_id}', status={status}. "
                f"Common causes:\n"
                f"  - HIXL library not installed or incompatible version\n"
                f"  - RDMA hardware not available on this node\n"
                f"  - CANN driver/runtime version mismatch"
            )

        self._hixl_initialized = True
        logger.info(
            f"HIXL engine initialized with local_engine_id="
            f"{self._local_engine_id}"
        )
```

### 4.6 `actor_has_tensor_transport`

```python
    def actor_has_tensor_transport(self, actor: "ray.actor.ActorHandle") -> bool:
        """Check if a remote actor has the HIXL transport available."""
        # TODO: This is called on a .remote RDT call, so it's quite expensive.
        def __ray_actor_has_tensor_transport__(
            self: "ray.actor.ActorHandle",
        ) -> bool:
            # Check if hixl_wrapper is installed and can initialize
            try:
                from ray.experimental.rdt.util import (
                    get_tensor_transport_manager,
                )

                get_tensor_transport_manager("HIXL")._ensure_hixl_initialized()
                return True
            except Exception:
                return False

        return ray.get(
            actor.__ray_call__.options(concurrency_group="_ray_system").remote(
                __ray_actor_has_tensor_transport__
            )
        )
```

### 4.7 `register_hixl_memory`

```python
    def register_hixl_memory(self, tensor: "torch.Tensor") -> None:
        """Registers the tensor's memory with HIXL and bumps the reference
        count so the memory region is never deregistered.

        Mirrors NixlTensorTransport.register_nixl_memory().
        Call this to pre-register a tensor's memory for the lifetime of the
        process, which can improve performance if the same tensor is re-used
        in multiple RDT objects.
        """
        self._add_tensor_descs([tensor])
```

### 4.8 `deregister_hixl_memory`

```python
    def deregister_hixl_memory(self, tensor: "torch.Tensor") -> None:
        """Decrements the reference count for the tensor's HIXL memory
        registration added by register_hixl_memory.

        If the reference count reaches 0, the memory is deregistered from
        HIXL. This should only be called after register_hixl_memory has been
        called for this tensor. Any existing ObjectRef instances that reference
        this tensor's memory will keep the HIXL registration alive independently
        until they go out of scope.

        Mirrors NixlTensorTransport.deregister_nixl_memory().
        """
        self._remove_tensor_descs([tensor])
```

### 4.9 `extract_tensor_transport_metadata`

```python
    def extract_tensor_transport_metadata(
        self,
        obj_id: str,
        rdt_object: List["torch.Tensor"],
    ) -> HixlTransportMetadata:
        """Source side: register tensor memory and serialize descriptors.

        Called on the source actor immediately after the task creates the
        result tensors. We:
          1. Synchronize the device to ensure data is written.
          2. Register each tensor's storage with HIXL (RegisterMem).
          3. Serialize the memory descriptions as pickle bytes.
          4. Return HixlTransportMetadata with the serialized descs, the
             local engine ID, and the current meta version.

        Args:
            obj_id: The object ID for the RDT object.
            rdt_object: The RDT object (list of tensors).

        Returns:
            HixlTransportMetadata containing serialized memory descriptions
            and the local engine ID.
        """
        import torch

        with self._cache_lock:
            device = None
            tensor_meta = []
            mem_descs_for_serialization = []

            if rdt_object:
                # All tensors must share the same device type,
                # but we don't assume they're all on the same device index.
                devices = set()
                device = rdt_object[0].device
                for t in rdt_object:
                    if t.device.type != device.type:
                        raise ValueError(
                            "All tensors in an RDT object must have the same "
                            "device type."
                        )
                    if not t.is_contiguous():
                        raise ValueError(
                            "All tensors in an RDT object must be contiguous."
                        )
                    tensor_meta.append((t.shape, t.dtype))
                    devices.add(t.device)

                if device.type == "npu":
                    # Synchronize before registration to assure the data has
                    # been written — HIXL does not guarantee this.
                    for dev in devices:
                        torch.npu.synchronize(dev)

                self._add_tensor_descs(rdt_object)

                # Build serialization payload: for each registered tensor,
                # we pack (data_ptr, nbytes, mem_type_str). The receiver
                # uses these to construct TransferOpDesc tuples.
                for t in rdt_object:
                    key = t.untyped_storage().data_ptr()
                    desc = self._tensor_desc_cache[key]
                    mem_descs_for_serialization.append(
                        (key, desc.nbytes, desc.mem_type_str)
                    )

                serialized_mem_descs = pickle.dumps(mem_descs_for_serialization)
                engine_id = self._local_engine_id
                engine_meta_version = self._hixl_engine_meta_version
            else:
                serialized_mem_descs = None
                engine_id = None
                engine_meta_version = None

            ret = HixlTransportMetadata(
                tensor_meta=tensor_meta,
                tensor_device=device.type if device else None,
                hixl_serialized_mem_descs=serialized_mem_descs,
                hixl_engine_id=engine_id,
                hixl_engine_meta_version=engine_meta_version,
            )
            self._put_meta(obj_id, ret)
            return ret
```

### 4.10 `get_communicator_metadata`

```python
    def get_communicator_metadata(
        self,
        src_actor: "ray.actor.ActorHandle",
        dst_actor: "ray.actor.ActorHandle",
        backend: Optional[str] = None,
    ) -> HixlCommunicatorMetadata:
        """One-sided RDMA transport: no communicator metadata needed."""
        return HixlCommunicatorMetadata()
```

### 4.11 `fetch_multiple_tensors`

```python
    def fetch_multiple_tensors(
        self,
        obj_id: str,
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
        target_buffers: Optional[List["torch.Tensor"]] = None,
    ) -> HixlFetchRequest:
        """Receiver side: initiate an RDMA READ transfer.

        This triggers the transfer but does not wait for completion. Call
        wait_fetch_complete(fetch_request) to retrieve the tensors.

        Steps:
          1. Allocate target tensors (or use provided buffers).
          2. Register target memory with HIXL.
          3. Deserialize the source memory descriptions from metadata.
          4. Connect to the remote HIXL engine (using engine_id from metadata).
          5. Build TransferOpDesc tuples: (local_addr, remote_addr, len).
          6. Call hixl_wrapper.transfer_async("READ", op_descs, remote_engine_id).
          7. Return HixlFetchRequest with the async transfer handle.

        Args:
            obj_id: The object ID for the transfer.
            tensor_transport_metadata: Source-side metadata containing
                serialized memory descriptions and the remote engine ID.
            communicator_metadata: Empty HixlCommunicatorMetadata.
            target_buffers: Optional pre-allocated buffers to receive into.

        Returns:
            HixlFetchRequest carrying the async transfer state.
        """
        from ray.experimental.rdt.util import (
            create_empty_tensors_from_metadata,
        )

        tensors = target_buffers or create_empty_tensors_from_metadata(
            tensor_transport_metadata
        )

        assert isinstance(tensor_transport_metadata, HixlTransportMetadata)
        assert isinstance(communicator_metadata, HixlCommunicatorMetadata)

        serialized_mem_descs = tensor_transport_metadata.hixl_serialized_mem_descs
        remote_engine_id = tensor_transport_metadata.hixl_engine_id

        with self._aborted_transfer_obj_ids_lock:
            if obj_id in self._aborted_transfer_obj_ids:
                self._aborted_transfer_obj_ids.remove(obj_id)
                raise RuntimeError(
                    f"HIXL transfer aborted for object id: {obj_id}"
                )

        transfer_req = None
        added_tensor_descs = False

        assert tensors

        try:
            self._ensure_hixl_initialized()

            # Register local target tensors with HIXL.
            self._add_tensor_descs(tensors)
            added_tensor_descs = True

            # Deserialize the source-side memory descriptions.
            remote_mem_descs = pickle.loads(serialized_mem_descs)

            # Connect to the remote HIXL engine (or reuse cached connection).
            remote_engine_meta_version = (
                tensor_transport_metadata.hixl_engine_meta_version
            )

            self._connect_remote_engine(
                remote_engine_id, remote_engine_meta_version
            )

            # Build TransferOpDesc tuples for RDMA READ.
            # For each tensor pair (local target, remote source):
            #   local_addr  = target tensor's storage data_ptr
            #   remote_addr = source tensor's data_ptr (from deserialized mem desc)
            #   len         = nbytes (must match; we validate this)
            op_descs = []
            for i, t in enumerate(tensors):
                remote_addr, remote_nbytes, _ = remote_mem_descs[i]
                local_addr = t.untyped_storage().data_ptr()
                local_nbytes = t.untyped_storage().nbytes()
                if local_nbytes != remote_nbytes:
                    raise RuntimeError(
                        f"HIXL transfer size mismatch for tensor {i}: "
                        f"local={local_nbytes} bytes vs remote={remote_nbytes} bytes"
                    )
                op_descs.append((local_addr, remote_addr, remote_nbytes))

            # Initiate async RDMA READ from remote engine.
            status, transfer_req = hixl_wrapper.transfer_async(
                remote_engine_id, "READ", op_descs
            )

            if status != hixl_wrapper.kSuccess:
                raise RuntimeError(
                    f"HIXL TransferAsync returned error status={status} "
                    f"for object id: {obj_id}"
                )

            return HixlFetchRequest(
                obj_id=obj_id,
                tensors=tensors,
                transfer_req=transfer_req,
                remote_engine_id=remote_engine_id,
                remove_tensor_descs=added_tensor_descs,
                transport=self,
            )
        except Exception:
            self._cleanup_transfer(
                obj_id, tensors, transfer_req, remote_engine_id,
                added_tensor_descs,
            )
            # Import here to avoid circular dependency on startup.
            from ray.exceptions import RayDirectTransportError

            raise RayDirectTransportError(
                f"The HIXL transfer failed for object id: {obj_id}. "
                f"The source actor may have died during the transfer. "
                f"The exception thrown from HIXL transfer was:\n "
                f"{traceback.format_exc()}"
            ) from None
```

> **关键差异**：NIXL 的 `fetch_multiple_tensors` 流程是：
> 1. `nixl_agent.deserialize_descs(serialized_descs)` → 得到远端描述
> 2. `_add_tensor_descs(tensors)` → 注册本地目标内存
> 3. `nixl_agent.get_xfer_descs(tensors)` → 得到本地描述
> 4. `nixl_agent.add_remote_agent(remote_meta)` → 添加远端 agent
> 5. `nixl_agent.initialize_xfer("READ", local, remote, remote_name, UUID)` → 初始化传输
> 6. `nixl_agent.transfer(xfer_handle)` → 启动传输
>
> HIXL 的流程是：
> 1. `pickle.loads(serialized_mem_descs)` → 得到远端 `(data_ptr, nbytes, mem_type_str)` 列表
> 2. `_add_tensor_descs(tensors)` → 注册本地目标内存
> 3. 手动构建 `[(local_addr, remote_addr, len)]` → 构造 TransferOpDesc
> 4. `_connect_remote_engine(remote_engine_id, version)` → 显式建链（替代 add_remote_agent）
> 5. `hixl_wrapper.transfer_async(remote_engine_id, "READ", op_descs)` → 启动传输（一步完成初始化+启动）

### 4.12 `wait_fetch_complete`

```python
    def wait_fetch_complete(
        self, fetch_request: FetchRequest, timeout: float = -1
    ) -> List["torch.Tensor"]:
        """Wait for a previously initiated HIXL fetch to complete.

        Polls hixl_wrapper.get_transfer_status until the state is "COMPLETED",
        "TIMEOUT", or "FAILED". Supports abort via _aborted_transfer_obj_ids.

        Args:
            fetch_request: The HixlFetchRequest returned by
                fetch_multiple_tensors.
            timeout: Maximum time in seconds to wait. -1 means wait
                indefinitely. 0 means return immediately if not ready.

        Returns:
            List of tensors that were transferred.

        Raises:
            RayDirectTransportError: If the transfer failed.
            TimeoutError: If the timeout is exceeded.
        """
        assert isinstance(fetch_request, HixlFetchRequest)
        obj_id = fetch_request.obj_id

        if not fetch_request.tensors:
            return fetch_request.tensors

        try:
            # Poll transfer status until completion.
            deadline = None if timeout < 0 else time.monotonic() + timeout
            while True:
                self._ensure_hixl_initialized()
                status, transfer_status = hixl_wrapper.get_transfer_status(
                    fetch_request.transfer_req
                )
                if status != hixl_wrapper.kSuccess:
                    raise RuntimeError(
                        f"HIXL GetTransferStatus returned error status={status} "
                        f"for object id: {obj_id}"
                    )

                if transfer_status == "FAILED":
                    raise RuntimeError(
                        f"HIXL transfer got FAILED state for object id: {obj_id}"
                    )
                if transfer_status == "TIMEOUT":
                    raise RuntimeError(
                        f"HIXL transfer got TIMEOUT state for object id: {obj_id}"
                    )
                if transfer_status == "WAITING":
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"HIXL transfer timed out after {timeout}s "
                            f"for object id: {obj_id}"
                        )
                    with self._aborted_transfer_obj_ids_lock:
                        if obj_id in self._aborted_transfer_obj_ids:
                            self._aborted_transfer_obj_ids.remove(obj_id)
                            raise RuntimeError(
                                f"HIXL transfer aborted for object id: {obj_id}"
                            )
                    time.sleep(0.001)  # Avoid busy waiting
                elif transfer_status == "COMPLETED":
                    break

            return fetch_request.tensors
        except TimeoutError:
            raise
        except Exception:
            from ray.exceptions import RayDirectTransportError

            raise RayDirectTransportError(
                f"The HIXL transfer failed for object id: {obj_id}. "
                f"The source actor may have died during the transfer. "
                f"The exception thrown from HIXL transfer was:\n "
                f"{traceback.format_exc()}"
            ) from None
```

### 4.13 `_cleanup_transfer`

```python
    def _cleanup_transfer(
        self,
        obj_id: str,
        tensors: List["torch.Tensor"],
        transfer_req: Any,
        remote_engine_id: Optional[str],
        remove_tensor_descs: bool,
    ) -> None:
        """Best-effort cleanup after a transfer completes or fails.

        We may encounter errors or HIXL may raise errors like connection
        loss, so we do best-effort cleanup without raising further errors.
        """
        if not self._hixl_initialized:
            return

        with self._aborted_transfer_obj_ids_lock:
            self._aborted_transfer_obj_ids.discard(obj_id)

        # HIXL does not have an explicit release_xfer_handle API;
        # the TransferReq is consumed by GetTransferStatus polling.

        # Evict remote engine from LRU cache if caching is disabled.
        if HIXL_REMOTE_ENGINE_CACHE_MAXSIZE == 0 and remote_engine_id:
            self._disconnect_remote_engine(remote_engine_id)

        if remove_tensor_descs:
            self._remove_tensor_descs(tensors)
```

### 4.14 `recv_multiple_tensors`

```python
    def recv_multiple_tensors(
        self,
        obj_id: str,
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
        target_buffers: Optional[List["torch.Tensor"]] = None,
    ) -> List["torch.Tensor"]:
        """Receives multiple tensors synchronously (fetch + wait)."""
        fetch_request = self.fetch_multiple_tensors(
            obj_id, tensor_transport_metadata, communicator_metadata,
            target_buffers,
        )
        return self.wait_fetch_complete(fetch_request)
```

### 4.15 `send_multiple_tensors`

```python
    def send_multiple_tensors(
        self,
        tensors: List["torch.Tensor"],
        tensor_transport_metadata: TensorTransportMetadata,
        communicator_metadata: CommunicatorMetadata,
    ):
        """Not implemented — HIXL is a one-sided transport."""
        raise NotImplementedError(
            "HIXL transport does not support send_multiple_tensors, "
            "since it is a one-sided transport."
        )
```

### 4.16 `garbage_collect`

```python
    def garbage_collect(
        self,
        obj_id: str,
        tensor_transport_meta: TensorTransportMetadata,
        tensors: List["torch.Tensor"],
    ):
        """Release source-side resources for an RDT object.

        Called on the source actor after Ray's distributed ref counting
        determines the object is out of scope. We:
          1. Pop the metadata from _managed_meta_hixl.
          2. Remove tensor descriptors (decrement ref count; deregister
             when it reaches zero).
        """
        with self._cache_lock:
            assert isinstance(tensor_transport_meta, HixlTransportMetadata)
            if obj_id not in self._managed_meta_hixl:
                return
            self._managed_meta_hixl.pop(obj_id, None)
            self._remove_tensor_descs(tensors)
```

### 4.17 `abort_transport`

```python
    def abort_transport(
        self,
        obj_id: str,
        communicator_metadata: CommunicatorMetadata,
    ):
        """Mark a transfer as aborted so wait_fetch_complete can exit."""
        with self._aborted_transfer_obj_ids_lock:
            self._aborted_transfer_obj_ids.add(obj_id)
```

### 4.18 `_add_tensor_descs`

```python
    def _add_tensor_descs(self, tensors: List["torch.Tensor"]):
        """Register tensor memory with HIXL and bump reference counts.

        If a tensor's storage is already registered (keyed by data_ptr), we
        only increment the metadata_count. Otherwise we call
        hixl_wrapper.register_mem and cache the handle + registration params.
        """
        self._ensure_hixl_initialized()

        with self._cache_lock:
            for tensor in tensors:
                key = tensor.untyped_storage().data_ptr()
                if key in self._tensor_desc_cache:
                    self._tensor_desc_cache[key].metadata_count += 1
                    continue

                # Determine memory type: NPU tensors → "npu", CPU → "cpu".
                mem_type_str = "npu" if tensor.device.type == "npu" else "cpu"

                # Register the full underlying storage with HIXL.
                addr = tensor.untyped_storage().data_ptr()
                nbytes = tensor.untyped_storage().nbytes()

                try:
                    status, mem_handle = hixl_wrapper.register_mem(
                        (addr, nbytes), mem_type_str
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to register {mem_type_str} memory with HIXL "
                        f"(addr=0x{addr:x}, size={nbytes} bytes). "
                        f"Common causes:\n"
                        f"  - CANN driver/runtime not installed\n"
                        f"  - RDMA device not available\n"
                        f"  - HCCS link not established\n"
                        f"  - Container privilege level too low"
                    ) from e

                if status != hixl_wrapper.kSuccess:
                    raise RuntimeError(
                        f"HIXL RegisterMem returned error status={status} "
                        f"for {mem_type_str} memory "
                        f"(addr=0x{addr:x}, size={nbytes} bytes)"
                    )

                self._tensor_desc_cache[key] = HixlTensorDesc(
                    mem_handle=mem_handle,
                    nbytes=nbytes,
                    mem_type_str=mem_type_str,
                    metadata_count=1,
                )
```

### 4.19 `_remove_tensor_descs`

```python
    def _remove_tensor_descs(self, tensors: List["torch.Tensor"]):
        """Decrement reference counts and deregister when they reach zero.

        When metadata_count drops to zero we call hixl_wrapper.deregister_mem
        with the cached MemHandle and bump _hixl_engine_meta_version.
        """
        with self._cache_lock:
            for tensor in tensors:
                key = tensor.untyped_storage().data_ptr()
                if key not in self._tensor_desc_cache:
                    continue
                tensor_desc = self._tensor_desc_cache[key]
                tensor_desc.metadata_count -= 1
                if tensor_desc.metadata_count == 0:
                    self._tensor_desc_cache.pop(key)
                    try:
                        status = hixl_wrapper.deregister_mem(
                            tensor_desc.mem_handle
                        )
                        if status != hixl_wrapper.kSuccess:
                            logger.warning(
                                f"HIXL DeregisterMem returned status={status} "
                                f"for handle={tensor_desc.mem_handle}"
                            )
                    except Exception:
                        logger.warning(
                            f"HIXL DeregisterMem raised exception for "
                            f"handle={tensor_desc.mem_handle}",
                            exc_info=True,
                        )
                    self._hixl_engine_meta_version += 1
```

### 4.20 `_tensor_memory_registered`

```python
    def _tensor_memory_registered(self, t: "torch.Tensor") -> bool:
        """Check if the tensor's memory has been registered with HIXL."""
        entry = self._tensor_desc_cache.get(t.untyped_storage().data_ptr())
        return entry is not None
```

> **注意**：NIXL 的 `_tensor_memory_registered` 检查 `entry is not None and entry.reg_desc is not None`（区分 pool-managed 和传统注册）。HIXL 没有 pool 概念（初始实现不含 MemoryPool），所以只要 `entry is not None` 就足够了。如果后续加 MemoryPool，需要像 NIXL 那样检查 `mem_handle is not None`。

### 4.21 `_get_meta` / `_put_meta`

```python
    def _get_num_managed_meta_hixl(self) -> int:
        """Return the number of tracked HixlTransportMetadata objects."""
        with self._cache_lock:
            return len(self._managed_meta_hixl)

    def _get_meta(self, object_id: str) -> Optional[HixlTransportMetadata]:
        """Get the HIXL transport metadata for the given object ID."""
        with self._cache_lock:
            if object_id in self._managed_meta_hixl:
                return self._managed_meta_hixl[object_id]
            return None

    def _put_meta(self, object_id: str, meta: HixlTransportMetadata):
        """Store the HIXL transport metadata for the given object ID."""
        with self._cache_lock:
            self._managed_meta_hixl[object_id] = meta
```

### 4.22 远端引擎连接管理（LRU 缓存）

```python
    def _connect_remote_engine(
        self, remote_engine_id: str, remote_engine_meta_version: int
    ) -> None:
        """Connect to a remote HIXL engine, with LRU caching.

        Mirrors NixlTensorTransport's _remote_agents logic:
          - If the remote engine is already cached and the meta version
            matches, we reuse the connection (move to end of LRU).
          - If the meta version differs (source deregistered memory), we
            disconnect first and reconnect.
          - If the cache is full, evict the least recently used engine.
          # 情况 1：已在缓存 + 版号一致 → 复用连接，return，不 connect
          # 情况 2：已在缓存 + 版号不一致 → 断开 + 重连 + 存缓存
          # 情况 3：不在缓存 + 缓存未满 → connect + 存缓存
          # 情况 4：不在缓存 + 缓存已满 → 淘汰最旧 + connect + 存缓存
          # ===== else 分支（缓存关闭）=====
          # 情况只有一种：直接 connect，不查缓存，不存缓存
        """
        if HIXL_REMOTE_ENGINE_CACHE_MAXSIZE > 0:
            if remote_engine_id in self._remote_engines:
                cached_version = self._remote_engines[remote_engine_id]
                if cached_version != remote_engine_meta_version:
                    # Source deregistered memory — stale descriptors.
                    # Disconnect before reconnecting.
                    self._disconnect_remote_engine(remote_engine_id)
                else:
                    # Reuse cached connection; move to end of LRU.
                    self._remote_engines.move_to_end(remote_engine_id)
                    return

            elif len(self._remote_engines) >= HIXL_REMOTE_ENGINE_CACHE_MAXSIZE:
                # Evict least recently used remote engine.
                evicted_engine_id, _ = self._remote_engines.popitem(last=False)
                self._disconnect_remote_engine(evicted_engine_id)

            # Establish new connection.
            status = hixl_wrapper.connect(remote_engine_id)
            if status != hixl_wrapper.kSuccess and status != hixl_wrapper.kAlreadyConnected:
                raise RuntimeError(
                    f"HIXL Connect to '{remote_engine_id}' failed, "
                    f"status={status}"
                )

            self._remote_engines[remote_engine_id] = remote_engine_meta_version
        else:
            # No caching — connect fresh each time.
            status = hixl_wrapper.connect(remote_engine_id)
            if status != hixl_wrapper.kSuccess and status != hixl_wrapper.kAlreadyConnected:
                raise RuntimeError(
                    f"HIXL Connect to '{remote_engine_id}' failed, "
                    f"status={status}"
                )

    def _disconnect_remote_engine(self, remote_engine_id: str) -> None:
        """Disconnect from a remote HIXL engine (best-effort)."""
        try:
            hixl_wrapper.disconnect(remote_engine_id)
        except Exception:
            logger.warning(
                f"HIXL Disconnect from '{remote_engine_id}' raised exception",
                exc_info=True,
            )
```

## 6. 导入和模块结构

```python
import logging
import pickle
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import ray
from ray.experimental.rdt.tensor_transport_manager import (
    CommunicatorMetadata,
    FetchRequest,
    TensorTransportManager,
    TensorTransportMetadata,
)

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Lazy import: hixl_wrapper may not be installed on all nodes.
try:
    import hixl_wrapper
except ImportError:
    hixl_wrapper = None

# Maximum number of cached HIXL remote engine connections.
# When exceeded, the least recently used remote engine is evicted and
# Disconnect is called. Set to 0 to disable remote engine reuse.
HIXL_REMOTE_ENGINE_CACHE_MAXSIZE = 1000
```

> **导入差异**：NIXL 从 `ray._private.ray_constants` 导入 `NIXL_REMOTE_AGENT_CACHE_MAXSIZE`。HIXL 在模块内定义 `HIXL_REMOTE_ENGINE_CACHE_MAXSIZE = 1000`，因为这是 ray-ascend 的自定义常量，不应侵入 Ray 主仓库。后续如果需要，可以移到 `ray_ascend` 的配置模块中。

---

## 7. 注册方式

在 `ray-ascend/ray_ascend/__init__.py` 中已有 `register_hixl_tensor_transport` 函数（已经实现，只需确保导入路径正确）：

```python
def register_hixl_tensor_transport(devices: List[str] = ["npu", "cpu"]) -> None:
    import torch
    from ray.experimental import register_tensor_transport
    from ray_ascend.direct_transport.hixl_tensor_transport import HixlTensorTransport

    # Verify hixl_wrapper is importable before registration.
    try:
        import hixl_wrapper
    except ImportError as e:
        raise ImportError(
            "hixl_wrapper module not found. HIXL tensor transport requires "
            "the HIXL Engine wheel. Please install: "
            "pip install hixl_engine-0.0.1-py3-none-any.whl"
        ) from e

    register_tensor_transport("HIXL", devices, HixlTensorTransport, torch.Tensor)
```