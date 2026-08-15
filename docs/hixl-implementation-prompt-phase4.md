# Phase 4 Prompt：测试 + 调试

你需要在 ray-ascend 项目中为 HIXL tensor transport 创建完整的单元测试文件，覆盖所有核心方法的逻辑正确性。

## 背景

- HIXL tensor transport 目前**完全没有测试文件**
- `tests/direct_transport/` 下只有 `test_yr_transport.py`、`test_yr_transport_util.py` 和 `test_hccl_tensor_transport.py`
- HIXL 依赖 NPU 硬件和 RDMA 链路，测试需要分层：L1（纯 mock 单元测试，无硬件）和 L2（硬件集成测试，skipif 标记）
- Phase 3 已完成基类继承修复和命名统一（`agent → engine`），测试中应使用 `hixl_engine_meta_version` 等命名

## 参考：现有测试模式

请严格参考以下现有代码的测试风格和模式：

1. **NIXL 引用计数单元测试风格**：参考 `ray/python/ray/tests/rdt/test_rdt_nixl.py` 第 434-527 行
   - 直接 `NixlTensorTransport()` 实例化（不需要 Ray 集群）
   - 验证 `metadata_count` 的增减逻辑
   - 验证 `_remove_tensor_descs` 在 `metadata_count == 0` 时 deregister
   - 验证 `_managed_meta_nixl` 的 pop 和清理逻辑

2. **NIXL register/deregister 测试风格**：参考 `ray/python/ray/tests/rdt/test_rdt_nixl.py` 第 577-609 行
   - `register_nixl_memory → deregister_nixl_memory` 流程
   - 验证 GC 后引用计数归零

## Mock 设计

### MockHixlWrapper（手动模拟类）

模拟 `hixl_wrapper` 模块的全部 API。这是一个**手写的类**（不是 MagicMock），因为 `HixlTensorTransport` 直接调用其方法名和属性名，MagicMock 无法保证方法名匹配。

```python
class MockHixlWrapper:
    """Simulates hixl_wrapper module for unit testing.

    Key design decisions:
    - Hand-written class (not MagicMock) so method names match real hixl_wrapper
    - Internal state tracking via dicts so tests can verify registration/connect behavior
    - Auto-progression: get_transfer_status advances WAITING → COMPLETED on first call
    """

    # Status codes — match real hixl_wrapper constants
    kSuccess = 0
    kAlreadyConnected = 103903
    kFailed = 503900
    kParamInvalid = 103900
    kTimeout = 103901
    kNotConnected = 103902

    _registered_mems: Dict[int, int] = {}   # addr → mem_handle
    _connected_engines: Dict[str, bool] = {} # engine_id → True
    _next_handle: int = 1
    _transfer_reqs: Dict[int, str] = {}      # req_id → status_str

    @classmethod
    def initialize(cls, engine_id: str, options: dict) -> int:
        """Initialize HIXL engine. Always succeeds in mock."""
        return cls.kSuccess

    @classmethod
    def register_mem(cls, mem_desc: tuple, mem_type: str) -> tuple:
        """Register memory region. Returns (kSuccess, handle_int)."""
        addr, nbytes = mem_desc
        handle = cls._next_handle
        cls._next_handle += 1
        cls._registered_mems[addr] = handle
        return (cls.kSuccess, handle)

    @classmethod
    def deregister_mem(cls, mem_handle: int) -> int:
        """Deregister memory region. Returns kSuccess."""
        cls._registered_mems = {
            k: v for k, v in cls._registered_mems.items() if v != mem_handle
        }
        return cls.kSuccess

    @classmethod
    def connect(cls, remote_engine: str, timeout_ms: int = 1000) -> int:
        """Connect to remote engine. Returns kSuccess."""
        cls._connected_engines[remote_engine] = True
        return cls.kSuccess

    @classmethod
    def disconnect(cls, remote_engine: str, timeout_ms: int = 1000) -> int:
        """Disconnect from remote engine. Returns kSuccess."""
        cls._connected_engines.pop(remote_engine, None)
        return cls.kSuccess

    @classmethod
    def transfer_async(cls, remote_engine: str, operation: str, op_descs: list) -> tuple:
        """Initiate async transfer. Returns (kSuccess, req_id_int)."""
        req_id = cls._next_handle
        cls._next_handle += 1
        cls._transfer_reqs[req_id] = "WAITING"
        return (cls.kSuccess, req_id)

    @classmethod
    def get_transfer_status(cls, req_id: int) -> tuple:
        """Poll transfer status. Auto-progresses WAITING → COMPLETED."""
        if req_id not in cls._transfer_reqs:
            return (cls.kFailed, "FAILED")
        current = cls._transfer_reqs[req_id]
        if current == "WAITING":
            cls._transfer_reqs[req_id] = "COMPLETED"
            return (cls.kSuccess, "WAITING")  # First poll returns WAITING
        return (cls.kSuccess, current)

    @classmethod
    def reset(cls):
        """Reset all mock state between tests."""
        cls._registered_mems.clear()
        cls._connected_engines.clear()
        cls._transfer_reqs.clear()
        cls._next_handle = 1
```

> **注意**：`get_transfer_status` 的 mock 需要模拟真实 HIXL 行为——第一次轮询返回 `"WAITING"`，随后自动推进到 `"COMPLETED"`。这样 `wait_fetch_complete` 的轮询循环可以正常退出，不需要 `time.sleep` 模拟。

### Patch 目标

```python
PATCH_TARGET = "ray_ascend.direct_transport.hixl_tensor_transport.hixl_wrapper"
```

这是 `hixl_tensor_transport.py` 第 20-22 行的 lazy import 变量名。patch 这个位置会让 `HixlTensorTransport` 内所有对 `hixl_wrapper.xxx` 的调用走 mock。

### Ray mock

`_ensure_hixl_initialized()` 调用 `ray.get_runtime_context()` 和 `ray.util.get_node_ip_address()`，需要 patch：

```python
with patch("ray.get_runtime_context") as mock_ctx, \
     patch("ray.util.get_node_ip_address", return_value="10.0.0.1"):
    mock_ctx.return_value.get_actor_id.return_value = "test_actor_123"
```

### Fixture 设计

```python
@pytest.fixture
def mock_hixl_wrapper():
    """Patch hixl_wrapper with a MockHixlWrapper instance."""
    mock = MockHixlWrapper()
    with patch(PATCH_TARGET, mock):
        mock.reset()
        yield mock

@pytest.fixture
def transport(mock_hixl_wrapper):
    """Create HixlTensorTransport with mocked hixl_wrapper and Ray runtime."""
    with (
        patch(PATCH_TARGET, mock_hixl_wrapper),
        patch("ray.get_runtime_context") as mock_ctx,
        patch("ray.util.get_node_ip_address", return_value="10.0.0.1"),
    ):
        mock_ctx.return_value.get_actor_id.return_value = "test_actor_123"
        t = HixlTensorTransport()
        t._ensure_hixl_initialized()
        yield t
```

> **注意**：`transport` fixture 中需要同时 patch `hixl_wrapper` 和 Ray API，且 `_ensure_hixl_initialized()` 在 fixture 内调用，这样后续测试中 transport 已处于初始化状态。

## 需要创建的测试文件

创建文件 `ray-ascend/tests/direct_transport/test_hixl_tensor_transport.py`，包含以下 11 个测试套件：

### Suite 1：TestDataClasses（纯 Python，无 mock）

验证数据类定义和继承关系：

```python
class TestDataClasses:
    """Verify data class definitions and inheritance."""

    def test_hixl_communicator_metadata_inherits(self):
        assert issubclass(HixlCommunicatorMetadata, CommunicatorMetadata)

    def test_hixl_transport_metadata_inherits(self):
        assert issubclass(HixlTransportMetadata, TensorTransportMetadata)

    def test_hixl_transport_metadata_fields(self):
        meta = HixlTransportMetadata(
            tensor_meta=[((2, 3), torch.float32)],
            tensor_device="npu",
            hixl_serialized_mem_descs=b"fake",
            hixl_engine_id="10.0.0.1:12345",
            hixl_engine_meta_version=0,
        )
        assert meta.hixl_serialized_mem_descs is not None
        assert meta.hixl_engine_id is not None
        assert meta.hixl_engine_meta_version == 0

    def test_hixl_transport_metadata_no_duplicate_base_fields(self):
        """子类不应重复定义基类的 tensor_meta 和 tensor_device 字段。
        dataclass 继承中如果子类重新定义了基类字段，会导致字段顺序错误。
        """
        base_fields = [f.name for f in TensorTransportMetadata.__dataclass_fields__.values()]
        child_fields = [f.name for f in HixlTransportMetadata.__dataclass_fields__.values()]
        # 基类字段应出现在子类字段列表的前面（继承顺序），但不应有重复定义
        # dataclass 继承的正确行为：子类字段列表 = 基类字段 + 新增字段
        new_fields = [f for f in child_fields if f not in base_fields]
        assert "hixl_serialized_mem_descs" in new_fields
        assert "hixl_engine_id" in new_fields
        assert "hixl_engine_meta_version" in new_fields

    def test_hixl_tensor_desc_fields(self):
        desc = HixlTensorDesc(mem_handle=42, nbytes=1024, mem_type_str="npu", metadata_count=1)
        assert desc.mem_handle == 42
        assert desc.nbytes == 1024
        assert desc.mem_type_str == "npu"
        assert desc.metadata_count == 1

    def test_hixl_fetch_request_inherits(self):
        assert issubclass(HixlFetchRequest, FetchRequest)

    def test_hixl_fetch_request_custom_fields(self):
        req = HixlFetchRequest(
            obj_id="test_obj",
            tensors=[],
            transfer_req=123,
            remote_engine_id="10.0.0.1:12345",
            remove_tensor_descs=True,
            transport=None,  # None so __del__ won't crash
        )
        assert req.transfer_req == 123
        assert req.remote_engine_id == "10.0.0.1:12345"
        assert req.remove_tensor_descs is True
```

### Suite 2：TestTransportProperties（纯 Python，无 mock）

```python
class TestTransportProperties:
    """Test static properties without hardware or mock."""

    def test_tensor_transport_backend(self):
        t = HixlTensorTransport()
        assert t.tensor_transport_backend() == "HIXL"

    def test_is_one_sided(self):
        assert HixlTensorTransport.is_one_sided() is True

    def test_can_abort_transport(self):
        assert HixlTensorTransport.can_abort_transport() is True

    def test_inherits_tensor_transport_manager(self):
        assert issubclass(HixlTensorTransport, TensorTransportManager)

    def test_send_multiple_tensors_raises(self):
        t = HixlTensorTransport()
        with pytest.raises(NotImplementedError, match="one-sided"):
            t.send_multiple_tensors([], HixlTransportMetadata(tensor_meta=[], tensor_device=None), HixlCommunicatorMetadata())
```

### Suite 3：TestMemoryRegistration（mock hixl_wrapper）

```python
class TestMemoryRegistration:
    """Test _add_tensor_descs and _remove_tensor_descs with mock."""

    def test_register_new_cpu_tensor(self, transport, mock_hixl_wrapper):
        """New CPU tensor → register_mem called, metadata_count=1, mem_type_str='cpu'."""
        t = torch.randn(2, 3, device="cpu")
        transport._add_tensor_descs([t])
        key = t.untyped_storage().data_ptr()
        assert key in transport._tensor_desc_cache
        desc = transport._tensor_desc_cache[key]
        assert desc.metadata_count == 1
        assert desc.mem_type_str == "cpu"
        assert desc.mem_handle in mock_hixl_wrapper._registered_mems.values()

    def test_register_same_tensor_twice_bumps_ref_count(self, transport, mock_hixl_wrapper):
        """Same tensor registered twice → metadata_count=2, no second register_mem call."""
        t = torch.randn(2, 3, device="cpu")
        transport._add_tensor_descs([t])
        transport._add_tensor_descs([t])
        key = t.untyped_storage().data_ptr()
        assert transport._tensor_desc_cache[key].metadata_count == 2
        # Only one register_mem call (check mock handle count)
        assert len(mock_hixl_wrapper._registered_mems) == 1

    def test_register_multiple_tensors(self, transport, mock_hixl_wrapper):
        """Multiple different tensors → each gets its own HixlTensorDesc."""
        t1 = torch.randn(2, 3, device="cpu")
        t2 = torch.randn(4, 5, device="cpu")
        transport._add_tensor_descs([t1, t2])
        assert len(transport._tensor_desc_cache) == 2

    def test_deregister_when_ref_count_zero(self, transport, mock_hixl_wrapper):
        """metadata_count→0 → deregister_mem called, entry removed from cache."""
        t = torch.randn(2, 3, device="cpu")
        transport._add_tensor_descs([t])
        key = t.untyped_storage().data_ptr()
        handle = transport._tensor_desc_cache[key].mem_handle

        transport._remove_tensor_descs([t])
        assert key not in transport._tensor_desc_cache
        assert handle not in mock_hixl_wrapper._registered_mems.values()

    def test_partial_deregister_keeps_registration(self, transport, mock_hixl_wrapper):
        """metadata_count 2→1 → entry stays, no deregister_mem call."""
        t = torch.randn(2, 3, device="cpu")
        transport._add_tensor_descs([t])
        transport._add_tensor_descs([t])  # ref_count = 2

        transport._remove_tensor_descs([t])  # ref_count = 1
        key = t.untyped_storage().data_ptr()
        assert key in transport._tensor_desc_cache
        assert transport._tensor_desc_cache[key].metadata_count == 1
        # Handle still in mock registered_mems
        assert len(mock_hixl_wrapper._registered_mems) == 1

    def test_deregister_unknown_tensor_is_noop(self, transport, mock_hixl_wrapper):
        """Removing a tensor not in cache should skip silently."""
        t = torch.randn(2, 3, device="cpu")
        transport._remove_tensor_descs([t])  # Never registered
        # Should not crash

    def test_engine_meta_version_bumps_on_full_deregister(self, transport, mock_hixl_wrapper):
        """_hixl_engine_meta_version increments when memory is fully deregistered."""
        initial = transport._hixl_engine_meta_version
        t = torch.randn(2, 3, device="cpu")
        transport._add_tensor_descs([t])
        transport._remove_tensor_descs([t])
        assert transport._hixl_engine_meta_version > initial

    def test_engine_meta_version_no_bump_on_partial_deregister(self, transport, mock_hixl_wrapper):
        """_hixl_engine_meta_version does NOT change when metadata_count > 0 after deregister."""
        initial = transport._hixl_engine_meta_version
        t = torch.randn(2, 3, device="cpu")
        transport._add_tensor_descs([t])
        transport._add_tensor_descs([t])
        transport._remove_tensor_descs([t])
        assert transport._hixl_engine_meta_version == initial

    def test_tensor_memory_registered(self, transport, mock_hixl_wrapper):
        """_tensor_memory_registered returns True for registered, False for unregistered."""
        t = torch.randn(2, 3, device="cpu")
        assert transport._tensor_memory_registered(t) is False
        transport._add_tensor_descs([t])
        assert transport._tensor_memory_registered(t) is True
```

### Suite 4：TestMetadataExtraction（mock hixl_wrapper）

```python
class TestMetadataExtraction:
    """Test extract_tensor_transport_metadata with mock."""

    def test_basic_extraction_cpu(self, transport, mock_hixl_wrapper):
        """CPU tensor extraction → registers memory, returns metadata with correct fields."""
        tensors = [torch.randn(2, 3, device="cpu")]
        meta = transport.extract_tensor_transport_metadata("obj1", tensors)
        assert isinstance(meta, HixlTransportMetadata)
        assert meta.tensor_device == "cpu"
        assert len(meta.tensor_meta) == 1
        assert meta.hixl_serialized_mem_descs is not None
        assert meta.hixl_engine_id == transport._local_engine_id
        assert meta.hixl_engine_meta_version == transport._hixl_engine_meta_version

    def test_metadata_stored_in_managed_meta(self, transport, mock_hixl_wrapper):
        """extract_ stores metadata in _managed_meta_hixl."""
        tensors = [torch.randn(2, 3, device="cpu")]
        meta = transport.extract_tensor_transport_metadata("obj1", tensors)
        assert transport._get_meta("obj1") == meta

    def test_serialized_mem_descs_format(self, transport, mock_hixl_wrapper):
        """Serialized descs = pickle([(data_ptr, nbytes, mem_type_str)])."""
        tensors = [torch.randn(2, 3, device="cpu")]
        meta = transport.extract_tensor_transport_metadata("obj1", tensors)
        descs = pickle.loads(meta.hixl_serialized_mem_descs)
        assert len(descs) == 1
        data_ptr, nbytes, mem_type = descs[0]
        assert mem_type == "cpu"
        assert nbytes == tensors[0].untyped_storage().nbytes()

    def test_multiple_tensors_serialization(self, transport, mock_hixl_wrapper):
        """Multiple tensors → multiple entries in serialized descs."""
        tensors = [torch.randn(2, 3, device="cpu"), torch.randn(4, device="cpu")]
        meta = transport.extract_tensor_transport_metadata("obj1", tensors)
        descs = pickle.loads(meta.hixl_serialized_mem_descs)
        assert len(descs) == 2

    def test_contiguous_check_raises(self, transport, mock_hixl_wrapper):
        """Non-contiguous tensor → ValueError."""
        t = torch.randn(2, 4, device="cpu").t()  # Transposed → non-contiguous
        with pytest.raises(ValueError, match="contiguous"):
            transport.extract_tensor_transport_metadata("obj1", [t])

    def test_empty_object_returns_none_fields(self, transport, mock_hixl_wrapper):
        """Empty rdt_object → metadata with None serialized fields."""
        meta = transport.extract_tensor_transport_metadata("obj1", [])
        assert meta.hixl_serialized_mem_descs is None
        assert meta.hixl_engine_id is None
        assert meta.hixl_engine_meta_version is None
        assert meta.tensor_meta == []
        assert meta.tensor_device is None

    def test_get_communicator_metadata(self, transport, mock_hixl_wrapper):
        """get_communicator_metadata returns empty HixlCommunicatorMetadata."""
        comm = transport.get_communicator_metadata(None, None)
        assert isinstance(comm, HixlCommunicatorMetadata)
```

### Suite 5：TestFetchAndWait（mock hixl_wrapper）

```python
class TestFetchAndWait:
    """Test fetch_multiple_tensors and wait_fetch_complete with mock."""

    def test_basic_fetch_and_wait(self, transport, mock_hixl_wrapper):
        """Complete flow: extract → fetch → wait → result."""
        src_tensors = [torch.randn(2, 3, device="cpu")]
        meta = transport.extract_tensor_transport_metadata("obj1", src_tensors)
        fetch_req = transport.fetch_multiple_tensors("obj1", meta, HixlCommunicatorMetadata())
        assert isinstance(fetch_req, HixlFetchRequest)
        assert fetch_req.transfer_req is not None
        assert fetch_req.remote_engine_id == meta.hixl_engine_id

        result = transport.wait_fetch_complete(fetch_req)
        assert len(result) == 1
        assert result[0].shape == src_tensors[0].shape

    def test_fetch_registers_target_tensors(self, transport, mock_hixl_wrapper):
        """Fetch should register target tensors' memory with HIXL."""
        src_tensors = [torch.randn(2, 3, device="cpu")]
        meta = transport.extract_tensor_transport_metadata("obj1", src_tensors)

        fetch_req = transport.fetch_multiple_tensors("obj1", meta, HixlCommunicatorMetadata())

        # Target tensors should be registered
        for t in fetch_req.tensors:
            key = t.untyped_storage().data_ptr()
            assert key in transport._tensor_desc_cache

    def test_fetch_connects_remote_engine(self, transport, mock_hixl_wrapper):
        """Fetch should connect to remote engine specified in metadata."""
        src_tensors = [torch.randn(2, 3, device="cpu")]
        meta = transport.extract_tensor_transport_metadata("obj1", src_tensors)

        fetch_req = transport.fetch_multiple_tensors("obj1", meta, HixlCommunicatorMetadata())
        remote_engine_id = meta.hixl_engine_id
        assert remote_engine_id in mock_hixl_wrapper._connected_engines

    def test_fetch_caches_remote_engine(self, transport, mock_hixl_wrapper):
        """Second fetch to same engine should reuse connection (LRU cache)."""
        src_tensors = [torch.randn(2, 3, device="cpu")]
        meta1 = transport.extract_tensor_transport_metadata("obj1", src_tensors)
        meta2 = transport.extract_tensor_transport_metadata("obj2", src_tensors)

        transport.fetch_multiple_tensors("obj1", meta1, HixlCommunicatorMetadata())
        transport.fetch_multiple_tensors("obj2", meta2, HixlCommunicatorMetadata())

        assert meta1.hixl_engine_id in transport._remote_engines

    def test_fetch_size_mismatch_raises(self, transport, mock_hixl_wrapper):
        """Local vs remote nbytes mismatch → RuntimeError."""
        src_tensors = [torch.randn(2, 3, device="cpu")]
        meta = transport.extract_tensor_transport_metadata("obj1", src_tensors)

        # Tamper with serialized descs to create size mismatch
        descs = pickle.loads(meta.hixl_serialized_mem_descs)
        descs[0] = (descs[0][0], descs[0][1] + 100, descs[0][2])  # wrong nbytes
        meta.hixl_serialized_mem_descs = pickle.dumps(descs)

        with pytest.raises(Exception, match="size mismatch"):
            transport.fetch_multiple_tensors("obj1", meta, HixlCommunicatorMetadata())

    def test_recv_multiple_tensors_is_sync_wrapper(self, transport, mock_hixl_wrapper):
        """recv_multiple_tensors = fetch + wait."""
        src_tensors = [torch.randn(2, 3, device="cpu")]
        meta = transport.extract_tensor_transport_metadata("obj1", src_tensors)

        result = transport.recv_multiple_tensors("obj1", meta, HixlCommunicatorMetadata())
        assert len(result) == 1
        assert result[0].shape == src_tensors[0].shape
```

### Suite 6：TestGarbageCollection（mock hixl_wrapper）

```python
class TestGarbageCollection:
    """Test garbage_collect with mock."""

    def test_gc_removes_meta_and_deregisters(self, transport, mock_hixl_wrapper):
        """GC pops metadata and deregisters tensor when ref_count → 0."""
        tensors = [torch.randn(2, 3, device="cpu")]
        meta = transport.extract_tensor_transport_metadata("obj1", tensors)
        transport.garbage_collect("obj1", meta, tensors)

        assert transport._get_meta("obj1") is None
        key = tensors[0].untyped_storage().data_ptr()
        assert key not in transport._tensor_desc_cache

    def test_gc_unknown_obj_id_is_noop(self, transport, mock_hixl_wrapper):
        """GC for unknown obj_id → returns without error."""
        meta = HixlTransportMetadata(
            tensor_meta=[], tensor_device=None,
            hixl_serialized_mem_descs=None,
        )
        transport.garbage_collect("unknown_obj", meta, [])
        # Should not raise

    def test_gc_shared_tensor_keeps_registration(self, transport, mock_hixl_wrapper):
        """GC of one metadata keeps tensor if another metadata still references it."""
        t = torch.randn(2, 3, device="cpu")
        meta1 = transport.extract_tensor_transport_metadata("obj1", [t])
        meta2 = transport.extract_tensor_transport_metadata("obj2", [t])

        transport.garbage_collect("obj1", meta1, [t])
        key = t.untyped_storage().data_ptr()
        assert key in transport._tensor_desc_cache
        assert transport._tensor_desc_cache[key].metadata_count == 1

    def test_gc_second_time_removes_registration(self, transport, mock_hixl_wrapper):
        """GC of both metadatas → tensor fully deregistered."""
        t = torch.randn(2, 3, device="cpu")
        meta1 = transport.extract_tensor_transport_metadata("obj1", [t])
        meta2 = transport.extract_tensor_transport_metadata("obj2", [t])

        transport.garbage_collect("obj1", meta1, [t])
        transport.garbage_collect("obj2", meta2, [t])
        key = t.untyped_storage().data_ptr()
        assert key not in transport._tensor_desc_cache

    def test_gc_bumps_engine_meta_version(self, transport, mock_hixl_wrapper):
        """GC that fully deregisters memory bumps _hixl_engine_meta_version."""
        initial = transport._hixl_engine_meta_version
        tensors = [torch.randn(2, 3, device="cpu")]
        meta = transport.extract_tensor_transport_metadata("obj1", tensors)
        transport.garbage_collect("obj1", meta, tensors)
        assert transport._hixl_engine_meta_version > initial
```

### Suite 7：TestAbortTransport（mock hixl_wrapper）

```python
class TestAbortTransport:
    """Test abort_transport mechanism."""

    def test_abort_marks_obj_id(self, transport, mock_hixl_wrapper):
        """abort_transport adds obj_id to _aborted_transfer_obj_ids."""
        transport.abort_transport("obj1", HixlCommunicatorMetadata())
        assert "obj1" in transport._aborted_transfer_obj_ids

    def test_aborted_fetch_raises_error(self, transport, mock_hixl_wrapper):
        """Fetch on aborted obj_id → RuntimeError with 'aborted' message."""
        transport.abort_transport("obj1", HixlCommunicatorMetadata())

        src_tensors = [torch.randn(2, 3, device="cpu")]
        meta = transport.extract_tensor_transport_metadata("obj1", src_tensors)

        with pytest.raises(RuntimeError, match="aborted"):
            transport.fetch_multiple_tensors("obj1", meta, HixlCommunicatorMetadata())

    def test_abort_removed_after_fetch_error(self, transport, mock_hixl_wrapper):
        """Aborted obj_id is removed from set after the RuntimeError is raised."""
        transport.abort_transport("obj1", HixlCommunicatorMetadata())
        src_tensors = [torch.randn(2, 3, device="cpu")]
        meta = transport.extract_tensor_transport_metadata("obj1", src_tensors)

        try:
            transport.fetch_multiple_tensors("obj1", meta, HixlCommunicatorMetadata())
        except RuntimeError:
            pass
        assert "obj1" not in transport._aborted_transfer_obj_ids
```

### Suite 8：TestRemoteEngineCache（mock hixl_wrapper）

```python
class TestRemoteEngineCache:
    """Test LRU remote engine connection caching."""

    def test_lru_eviction(self, transport, mock_hixl_wrapper):
        """When cache is full, least recently used engine is evicted."""
        import ray_ascend.direct_transport.hixl_tensor_transport as hixl_mod
        original = hixl_mod.HIXL_REMOTE_ENGINE_CACHE_MAXSIZE
        hixl_mod.HIXL_REMOTE_ENGINE_CACHE_MAXSIZE = 2

        try:
            transport._connect_remote_engine("engine_A", 0)
            transport._connect_remote_engine("engine_B", 0)
            transport._connect_remote_engine("engine_C", 0)  # evicts engine_A
            assert "engine_A" not in transport._remote_engines
            assert "engine_B" in transport._remote_engines
            assert "engine_C" in transport._remote_engines
        finally:
            hixl_mod.HIXL_REMOTE_ENGINE_CACHE_MAXSIZE = original

    def test_version_mismatch_reconnects(self, transport, mock_hixl_wrapper):
        """Different meta version → disconnect + reconnect with new version."""
        transport._connect_remote_engine("engine_A", 0)
        assert transport._remote_engines["engine_A"] == 0

        transport._connect_remote_engine("engine_A", 5)
        assert transport._remote_engines["engine_A"] == 5
        # Engine was reconnected (disconnect called then connect called)

    def test_version_match_reuses_connection(self, transport, mock_hixl_wrapper):
        """Same meta version → reuse cached connection, no new connect call."""
        transport._connect_remote_engine("engine_A", 0)
        initial_connected_count = len(mock_hixl_wrapper._connected_engines)

        transport._connect_remote_engine("engine_A", 0)
        # Should not call connect again
        assert len(mock_hixl_wrapper._connected_engines) == initial_connected_count

    def test_no_cache_mode_connects_fresh(self, transport, mock_hixl_wrapper):
        """HIXL_REMOTE_ENGINE_CACHE_MAXSIZE=0 → no caching, connect fresh each time."""
        import ray_ascend.direct_transport.hixl_tensor_transport as hixl_mod
        original = hixl_mod.HIXL_REMOTE_ENGINE_CACHE_MAXSIZE
        hixl_mod.HIXL_REMOTE_ENGINE_CACHE_MAXSIZE = 0

        try:
            transport._connect_remote_engine("engine_A", 0)
            assert len(transport._remote_engines) == 0
        finally:
            hixl_mod.HIXL_REMOTE_ENGINE_CACHE_MAXSIZE = original

    def test_disconnect_is_best_effort(self, transport, mock_hixl_wrapper):
        """_disconnect_remote_engine should not raise even if disconnect fails."""
        # Make disconnect raise
        mock_hixl_wrapper.disconnect = MagicMock(side_effect=RuntimeError("connection lost"))
        # Should not raise, just log warning
        transport._disconnect_remote_engine("engine_A")
```

### Suite 9：TestActorHealthCheck（mock Ray actor）

```python
class TestActorHealthCheck:
    """Test actor_has_tensor_transport with mock Ray actor."""

    def test_success(self, transport, mock_hixl_wrapper):
        mock_actor = MagicMock()
        mock_actor.__ray_call__ = MagicMock()
        mock_actor.__ray_call__.options.return_value = mock_actor.__ray_call__
        mock_actor.__ray_call__.remote.return_value = "mock_ref"

        with patch("ray.get", return_value=True):
            result = transport.actor_has_tensor_transport(mock_actor)
            assert result is True
            mock_actor.__ray_call__.options.assert_called_once_with(
                concurrency_group="_ray_system"
            )

    def test_failure(self, transport, mock_hixl_wrapper):
        mock_actor = MagicMock()
        mock_actor.__ray_call__ = MagicMock()
        mock_actor.__ray_call__.options.return_value = mock_actor.__ray_call__
        mock_actor.__ray_call__.remote.return_value = "mock_ref"

        with patch("ray.get", return_value=False):
            result = transport.actor_has_tensor_transport(mock_actor)
            assert result is False
```

### Suite 10：TestErrorHandling（mock hixl_wrapper）

```python
class TestErrorHandling:
    """Test error paths and edge cases."""

    def test_hixl_wrapper_not_installed_raises_import_error(self):
        """hixl_wrapper=None → _ensure_hixl_initialized raises ImportError."""
        with patch(PATCH_TARGET, None):
            t = HixlTensorTransport()
            with pytest.raises(ImportError, match="hixl_wrapper"):
                t._ensure_hixl_initialized()

    def test_register_mem_failure_raises_runtime_error(self, transport, mock_hixl_wrapper):
        """register_mem returns kFailed → RuntimeError."""
        mock_hixl_wrapper.register_mem = MagicMock(
            return_value=(mock_hixl_wrapper.kFailed, None)
        )
        t = torch.randn(2, 3, device="cpu")
        with pytest.raises(RuntimeError, match="RegisterMem"):
            transport._add_tensor_descs([t])

    def test_connect_failure_raises_runtime_error(self, transport, mock_hixl_wrapper):
        """connect returns kFailed → RuntimeError."""
        mock_hixl_wrapper.connect = MagicMock(return_value=mock_hixl_wrapper.kFailed)
        with pytest.raises(RuntimeError, match="Connect"):
            transport._connect_remote_engine("bad_engine", 0)

    def test_transfer_async_failure_raises_ray_direct_transport_error(self, transport, mock_hixl_wrapper):
        """transfer_async returns kFailed → RayDirectTransportError."""
        mock_hixl_wrapper.transfer_async = MagicMock(
            return_value=(mock_hixl_wrapper.kFailed, None)
        )
        src_tensors = [torch.randn(2, 3, device="cpu")]
        meta = transport.extract_tensor_transport_metadata("obj1", src_tensors)

        with pytest.raises(Exception, match="HIXL transfer failed"):
            transport.fetch_multiple_tensors("obj1", meta, HixlCommunicatorMetadata())

    def test_get_transfer_status_failed_state_raises(self, transport, mock_hixl_wrapper):
        """get_transfer_status returns FAILED → RuntimeError in wait_fetch_complete."""
        src_tensors = [torch.randn(2, 3, device="cpu")]
        meta = transport.extract_tensor_transport_metadata("obj1", src_tensors)
        fetch_req = transport.fetch_multiple_tensors("obj1", meta, HixlCommunicatorMetadata())

        # Override mock to return FAILED
        mock_hixl_wrapper.get_transfer_status = MagicMock(
            return_value=(mock_hixl_wrapper.kSuccess, "FAILED")
        )
        with pytest.raises(Exception, match="FAILED"):
            transport.wait_fetch_complete(fetch_req)

    def test_cleanup_on_fetch_failure_removes_target_descs(self, transport, mock_hixl_wrapper):
        """When fetch fails, target tensor descs should be cleaned up."""
        mock_hixl_wrapper.transfer_async = MagicMock(
            return_value=(mock_hixl_wrapper.kFailed, None)
        )
        src_tensors = [torch.randn(2, 3, device="cpu")]
        meta = transport.extract_tensor_transport_metadata("obj1", src_tensors)

        try:
            transport.fetch_multiple_tensors("obj1", meta, HixlCommunicatorMetadata())
        except Exception:
            pass

        # Source tensor should still be registered (from extract_)
        src_key = src_tensors[0].untyped_storage().data_ptr()
        assert src_key in transport._tensor_desc_cache

    def test_cleanup_without_hixl_initialized_is_noop(self):
        """_cleanup_transfer when _hixl_initialized=False → returns immediately."""
        t = HixlTensorTransport()
        # _hixl_initialized is False by default
        t._cleanup_transfer("obj1", [], None, None, False)
        # Should not raise

    def test_deregister_mem_failure_logs_warning(self, transport, mock_hixl_wrapper):
        """deregister_mem returning error → warning log, not exception."""
        mock_hixl_wrapper.deregister_mem = MagicMock(return_value=mock_hixl_wrapper.kFailed)
        t = torch.randn(2, 3, device="cpu")
        transport._add_tensor_descs([t])
        # Should not raise, just log
        transport._remove_tensor_descs([t])
```

### Suite 11：TestNPUIntegration（硬件依赖，skipif 标记）

```python
NPU_AVAILABLE = False
try:
    import torch_npu
    if torch.npu.is_available():
        NPU_AVAILABLE = True
except ImportError:
    pass

skip_no_npu = pytest.mark.skipif(not NPU_AVAILABLE, reason="NPU hardware not available")


@skip_no_npu
class TestNPUIntegration:
    """Integration tests requiring real NPU hardware and hixl_wrapper."""

    @pytest.fixture
    def transport_real(self):
        """Create HixlTensorTransport with real hixl_wrapper (no mock)."""
        pytest.importorskip("hixl_wrapper")
        import ray
        ray.init(ignore_reinit_error=True)
        t = HixlTensorTransport()
        t._ensure_hixl_initialized()
        yield t
        ray.shutdown()

    def test_npu_tensor_registration(self, transport_real):
        """Register and deregister a real NPU tensor."""
        t = torch.randn(2, 3, device="npu")
        transport_real._add_tensor_descs([t])
        key = t.untyped_storage().data_ptr()
        assert key in transport_real._tensor_desc_cache
        assert transport_real._tensor_desc_cache[key].mem_type_str == "npu"

        transport_real._remove_tensor_descs([t])
        assert key not in transport_real._tensor_desc_cache

    def test_npu_extract_metadata(self, transport_real):
        """Extract metadata for NPU tensors with real hixl_wrapper."""
        tensors = [torch.randn(2, 3, device="npu")]
        meta = transport_real.extract_tensor_transport_metadata("obj1", tensors)
        assert meta.tensor_device == "npu"
        assert meta.hixl_serialized_mem_descs is not None
```

## 输出要求

创建文件 `ray-ascend/tests/direct_transport/test_hixl_tensor_transport.py`，包含以上所有 11 个测试套件。

**关键约束**：
- L1 mock 测试全部使用 CPU tensor（`device="cpu"`），不需要 NPU
- NPU 测试只在 `TestNPUIntegration` 中，用 `pytest.mark.skipif` 标记
- 不要 mock torch — `untyped_storage().data_ptr()`、`nbytes()` 等是真实 Python 操作
- 创建测试用 `HixlFetchRequest` 时设 `transport=None`，防止 `__del__` 调用 cleanup
- LRU 缓存测试中临时修改 `HIXL_REMOTE_ENGINE_CACHE_MAXSIZE`，测试后恢复原值

## 验证标准

生成的测试文件应该能运行（假设依赖已安装）：

```bash
cd /home/lyy/code/hixl/ray-ascend

# L1 单元测试（不需要 NPU 硬件）
python -m pytest tests/direct_transport/test_hixl_tensor_transport.py -v -k "not NPUIntegration"

# 只跑数据类和静态属性（最快验证）
python -m pytest tests/direct_transport/test_hixl_tensor_transport.py::TestDataClasses -v
python -m pytest tests/direct_transport/test_hixl_tensor_transport.py::TestTransportProperties -v

# 覆盖率检查
python -m pytest tests/direct_transport/test_hixl_tensor_transport.py \
    --cov=ray_ascend.direct_transport.hixl_tensor_transport \
    --cov-report=term-missing \
    -k "not NPUIntegration"

# L2 集成测试（需要 NPU + RDMA 环境）
python -m pytest tests/direct_transport/test_hixl_tensor_transport.py::TestNPUIntegration -v
```
