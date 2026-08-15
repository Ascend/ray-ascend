import pytest

pytest.importorskip("hixl", reason="HIXL tests require the hixl wheel")
pytest.importorskip("torch_npu", reason="HIXL tests require torch_npu + NPU hardware")

import pickle

import ray
import torch

from ray_ascend import register_hixl_tensor_transport
from ray_ascend.direct_transport.hixl_tensor_transport import (
    HixlCommunicatorMetadata,
    HixlTensorTransport,
    HixlTransportMetadata,
)

register_hixl_tensor_transport(["npu", "cpu"])

DEFAULT_NPU_COUNT = 2


@pytest.fixture(scope="session")
def ray_cluster_with_npu():
    """Ray cluster with NPU resources. Mirrors tests/conftest.py, kept local."""
    if not torch.npu.is_available():
        pytest.skip("NPU hardware not available")
    if torch.npu.device_count() < DEFAULT_NPU_COUNT:
        pytest.skip(
            f"HIXL transfer tests need {DEFAULT_NPU_COUNT} NPU devices, "
            f"only {torch.npu.device_count()} available"
        )
    if not ray.is_initialized():
        try:
            ray.init(ignore_reinit_error=True, resources={"NPU": DEFAULT_NPU_COUNT})
        except ValueError:
            ray.init(ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture
def transport():
    """A bare HixlTensorTransport. teardown finalizes the engine if initialized."""
    t = HixlTensorTransport()
    yield t
    t.finalize()


class TestNpuMemoryRegistration:
    """Register/deregister NPU tensors against the real hixl engine."""

    @pytest.fixture(autouse=True)
    def _require_cluster(self, ray_cluster_with_npu):
        pass

    def test_register_new_npu_tensor(self, transport):
        transport._ensure_hixl_initialized()
        t = torch.randn(2, 3, device="npu")
        transport._add_tensor_descs([t])

        key = t.untyped_storage().data_ptr()
        assert key in transport._tensor_desc_cache
        desc = transport._tensor_desc_cache[key]
        assert desc.metadata_count == 1
        assert desc.mem_type_str == "npu"
        assert desc.mem_handle != 0

    def test_register_same_tensor_twice_bumps_ref_count(self, transport):
        transport._ensure_hixl_initialized()
        t = torch.randn(2, 3, device="npu")
        transport._add_tensor_descs([t])
        transport._add_tensor_descs([t])

        key = t.untyped_storage().data_ptr()
        assert transport._tensor_desc_cache[key].metadata_count == 2

    def test_partial_deregister_keeps_registration(self, transport):
        transport._ensure_hixl_initialized()
        t = torch.randn(2, 3, device="npu")
        transport._add_tensor_descs([t])
        transport._add_tensor_descs([t])

        transport._remove_tensor_descs([t])
        key = t.untyped_storage().data_ptr()
        assert key in transport._tensor_desc_cache
        assert transport._tensor_desc_cache[key].metadata_count == 1

    def test_full_deregister_bumps_meta_version(self, transport):
        transport._ensure_hixl_initialized()
        initial = transport._hixl_mem_generation

        t = torch.randn(2, 3, device="npu")
        transport._add_tensor_descs([t])
        transport._remove_tensor_descs([t])

        assert transport._hixl_mem_generation > initial

    def test_partial_deregister_does_not_bump_meta_version(self, transport):
        transport._ensure_hixl_initialized()
        initial = transport._hixl_mem_generation

        t = torch.randn(2, 3, device="npu")
        transport._add_tensor_descs([t])
        transport._add_tensor_descs([t])
        transport._remove_tensor_descs([t])
        assert transport._hixl_mem_generation == initial

    def test_tensor_memory_registered(self, transport):
        transport._ensure_hixl_initialized()
        t = torch.randn(2, 3, device="npu")
        assert transport._tensor_memory_registered(t) is False
        transport._add_tensor_descs([t])
        assert transport._tensor_memory_registered(t) is True


class TestMetadataExtraction:
    """extract_tensor_transport_metadata: register + serialize + store."""

    @pytest.fixture(autouse=True)
    def _require_cluster(self, ray_cluster_with_npu):
        pass

    def test_basic_extraction_npu(self, transport):
        transport._ensure_hixl_initialized()
        tensors = [torch.randn(2, 3, device="npu")]
        meta = transport.extract_tensor_transport_metadata("obj1", tensors)

        assert isinstance(meta, HixlTransportMetadata)
        assert meta.tensor_device == "npu"
        assert len(meta.tensor_meta) == 1
        assert meta.hixl_serialized_mem_descs is not None
        assert meta.hixl_engine_id == transport._local_engine_id
        assert meta.hixl_mem_generation == transport._hixl_mem_generation

    def test_metadata_stored_in_managed_meta(self, transport):
        transport._ensure_hixl_initialized()
        tensors = [torch.randn(2, 3, device="npu")]
        meta = transport.extract_tensor_transport_metadata("obj1", tensors)
        assert transport._get_meta("obj1") == meta

    def test_serialized_mem_descs_format(self, transport):
        transport._ensure_hixl_initialized()
        tensors = [torch.randn(2, 3, device="npu")]
        meta = transport.extract_tensor_transport_metadata("obj1", tensors)

        descs = pickle.loads(meta.hixl_serialized_mem_descs)
        assert len(descs) == 1
        _, nbytes, mem_type = descs[0]
        assert mem_type == "npu"
        assert nbytes == tensors[0].untyped_storage().nbytes()

    def test_multiple_tensors_serialization(self, transport):
        transport._ensure_hixl_initialized()
        tensors = [
            torch.randn(2, 3, device="npu"),
            torch.randn(4, device="npu"),
        ]
        meta = transport.extract_tensor_transport_metadata("obj1", tensors)
        descs = pickle.loads(meta.hixl_serialized_mem_descs)
        assert len(descs) == 2

    def test_contiguous_check_raises(self, transport):
        transport._ensure_hixl_initialized()
        t = torch.randn(2, 4, device="npu").t()
        with pytest.raises(ValueError, match="contiguous"):
            transport.extract_tensor_transport_metadata("obj1", [t])

    def test_empty_object_returns_none_fields(self, transport):
        transport._ensure_hixl_initialized()
        meta = transport.extract_tensor_transport_metadata("obj1", [])
        assert meta.hixl_serialized_mem_descs is None
        assert meta.hixl_engine_id is None
        assert meta.hixl_mem_generation is None
        assert meta.tensor_meta == []
        assert meta.tensor_device is None

    def test_get_communicator_metadata(self, transport):
        comm = transport.get_communicator_metadata(None, None)
        assert isinstance(comm, HixlCommunicatorMetadata)


class TestGarbageCollection:
    """garbage_collect: pop metadata, decrement ref count, deregister at zero."""

    @pytest.fixture(autouse=True)
    def _require_cluster(self, ray_cluster_with_npu):
        pass

    def test_gc_removes_meta_and_deregisters(self, transport):
        transport._ensure_hixl_initialized()
        tensors = [torch.randn(2, 3, device="npu")]
        meta = transport.extract_tensor_transport_metadata("obj1", tensors)

        transport.garbage_collect("obj1", meta, tensors)
        assert transport._get_meta("obj1") is None
        key = tensors[0].untyped_storage().data_ptr()
        assert key not in transport._tensor_desc_cache

    def test_gc_unknown_obj_id_is_noop(self, transport):
        transport._ensure_hixl_initialized()
        meta = HixlTransportMetadata(
            tensor_meta=[], tensor_device=None, hixl_serialized_mem_descs=None
        )
        transport.garbage_collect("unknown_obj", meta, [])  # must not raise

    def test_gc_shared_tensor_keeps_registration(self, transport):
        transport._ensure_hixl_initialized()
        t = torch.randn(2, 3, device="npu")
        meta1 = transport.extract_tensor_transport_metadata("obj1", [t])
        meta2 = transport.extract_tensor_transport_metadata("obj2", [t])

        transport.garbage_collect("obj1", meta1, [t])
        key = t.untyped_storage().data_ptr()
        assert key in transport._tensor_desc_cache
        assert transport._tensor_desc_cache[key].metadata_count == 1

    def test_gc_both_metadatas_fully_deregisters(self, transport):
        transport._ensure_hixl_initialized()
        t = torch.randn(2, 3, device="npu")
        meta1 = transport.extract_tensor_transport_metadata("obj1", [t])
        meta2 = transport.extract_tensor_transport_metadata("obj2", [t])

        transport.garbage_collect("obj1", meta1, [t])
        transport.garbage_collect("obj2", meta2, [t])
        key = t.untyped_storage().data_ptr()
        assert key not in transport._tensor_desc_cache


class TestRemoteEngineCache:
    """LRU eviction, version-mismatch reconnect, and reuse semantics."""

    @pytest.fixture(autouse=True)
    def _require_cluster(self, ray_cluster_with_npu):
        pass

    @pytest.fixture(autouse=True)
    def _stub_connect(self, transport, monkeypatch):
        """Replace the real RDMA connect/disconnect with no-op successes so
        only the cache logic is exercised."""
        transport._ensure_hixl_initialized()
        assert transport._hixl_engine is not None
        monkeypatch.setattr(
            transport._hixl_engine, "connect", lambda *a, **k: hixl.SUCCESS
        )
        monkeypatch.setattr(
            transport._hixl_engine, "disconnect", lambda *a, **k: hixl.SUCCESS
        )

    def test_version_match_reuses_connection(self, transport):
        local_id = transport._local_engine_id

        transport._connect_remote_engine(local_id, 0)
        assert local_id in transport._remote_engines
        size_after_first = len(transport._remote_engines)

        transport._connect_remote_engine(local_id, 0)
        assert len(transport._remote_engines) == size_after_first

    def test_version_mismatch_reconnects(self, transport):
        local_id = transport._local_engine_id

        transport._connect_remote_engine(local_id, 0)
        transport._connect_remote_engine(local_id, 5)
        assert transport._remote_engines[local_id] == 5

    def test_lru_eviction(self, transport, monkeypatch):
        import ray_ascend.direct_transport.hixl_tensor_transport as hixl_mod

        original = hixl_mod.HIXL_REMOTE_ENGINE_CACHE_MAXSIZE
        monkeypatch.setattr(hixl_mod, "HIXL_REMOTE_ENGINE_CACHE_MAXSIZE", 2)
        local_id = transport._local_engine_id

        try:
            ids = ["e_A", "e_B", local_id]
            for i, eid in enumerate(ids):
                transport._connect_remote_engine(eid, i)
            assert local_id in transport._remote_engines
        finally:
            monkeypatch.setattr(hixl_mod, "HIXL_REMOTE_ENGINE_CACHE_MAXSIZE", original)


@ray.remote(resources={"NPU": 1})
class _HixlHealthCheckActor:
    def __init__(self):
        register_hixl_tensor_transport(["npu", "cpu"])

    def health(self):
        from ray.experimental.rdt.util import get_tensor_transport_manager

        try:
            manager = get_tensor_transport_manager("HIXL")
            manager._ensure_hixl_initialized()
            return True
        except Exception:
            return False


@ray.remote(resources={"NPU": 1})
class _HixlSourceActor:
    def __init__(self):
        register_hixl_tensor_transport(["npu", "cpu"])

    @ray.method(tensor_transport="HIXL")
    def make_tensor(self):
        return torch.arange(12, dtype=torch.float32, device="npu").reshape(3, 4)

    def get_cache_state(self):
        """Introspect the actor-side HIXL cache for assertions."""
        import os

        import torch
        from ray.experimental.rdt.util import get_tensor_transport_manager

        mgr = get_tensor_transport_manager("HIXL")
        mgr._ensure_hixl_initialized()
        resolved = mgr._resolve_npu_device_id()
        try:
            current_device = torch.npu.current_device()
        except Exception as e:
            current_device = f"<err: {e}>"
        return {
            "driver_engine_id": mgr._local_engine_id,
            "tensor_desc_cache_size": len(mgr._tensor_desc_cache),
            "remote_engines_size": len(mgr._remote_engines),
            "hixl_initialized": mgr._hixl_initialized,
            "ascend_visible_devices": os.environ.get(
                "ASCEND_RT_VISIBLE_DEVICES", "<unset>"
            ),
            "resolved_device_id": resolved,
            "current_device": current_device,
        }


@ray.remote(resources={"NPU": 1})
class _HixlSinkActor:
    """Client side of the end-to-end transfer."""

    def __init__(self):
        register_hixl_tensor_transport(["npu", "cpu"])

    def recv_and_verify(self, ref):
        """Fetch the HIXL tensor referenced by `ref` via one-sided RDMA READ
        (runs in *this* sink process), assert it landed on the NPU with the
        right shape, then return the values as a CPU list for the driver to
        check. Returning CPU scalars avoids re-serializing the NPU tensor
        back through the object store (which would be a plain copy, not HIXL).
        """
        tensor = ray.get(ref)
        assert (
            tensor.device.type == "npu"
        ), f"fetched tensor not on npu, got {tensor.device}"
        assert tensor.shape == (3, 4), f"wrong shape {tuple(tensor.shape)}"
        return tensor.cpu().reshape(-1).tolist()

    def get_cache_state(self):
        """Same introspection as _HixlSourceActor, for the client-side cache."""
        import os

        import torch
        from ray.experimental.rdt.util import get_tensor_transport_manager

        mgr = get_tensor_transport_manager("HIXL")
        mgr._ensure_hixl_initialized()
        resolved = mgr._resolve_npu_device_id()
        try:
            current_device = torch.npu.current_device()
        except Exception as e:
            current_device = f"<err: {e}>"
        return {
            "driver_engine_id": mgr._local_engine_id,
            "tensor_desc_cache_size": len(mgr._tensor_desc_cache),
            "remote_engines_size": len(mgr._remote_engines),
            "hixl_initialized": mgr._hixl_initialized,
            "ascend_visible_devices": os.environ.get(
                "ASCEND_RT_VISIBLE_DEVICES", "<unset>"
            ),
            "resolved_device_id": resolved,
            "current_device": current_device,
        }


class TestEndToEndTransfer:
    """Full RDMA READ between two NPU actors."""

    @pytest.fixture(autouse=True)
    def _require_cluster(self, ray_cluster_with_npu):
        pass

    def test_tensor_transport_via_rdt(self):
        """HIXL-decorated remote method returns a tensor transported via HIXL."""
        source = _HixlSourceActor.remote()
        sink = _HixlSinkActor.remote()

        src_state = ray.get(source.get_cache_state.remote())
        dst_state = ray.get(sink.get_cache_state.remote())
        print(
            "\n[DIAG] source-actor: ascend_visible_devices="
            f"{src_state.get('ascend_visible_devices')} "
            f"resolved={src_state.get('resolved_device_id')} "
            f"current_device={src_state.get('current_device')} "
            f"engine_id={src_state.get('driver_engine_id')}"
        )
        print(
            "[DIAG] sink-actor: ascend_visible_devices="
            f"{dst_state.get('ascend_visible_devices')} "
            f"resolved={dst_state.get('resolved_device_id')} "
            f"current_device={dst_state.get('current_device')} "
            f"engine_id={dst_state.get('driver_engine_id')}"
        )

        ref = source.make_tensor.remote()
        got = ray.get(sink.recv_and_verify.remote(ref))

        expected = list(range(12))
        assert got == expected, f"HIXL transferred values wrong: {got}"

    def test_two_source_tensors_transferred(self):
        """Two sequential HIXL transfers reuse actor-side state. Verifies across
        two make_tensor calls:
          - Engine stays initialized (no re-init per transfer).
          - source tensor_desc_cache grows by one per transfer (distinct
            storages — expected RDT contract, not a leak).
          - sink remote engine cache stays at 1 (sole source -> reused
            connection).
        """
        source = _HixlSourceActor.remote()
        sink = _HixlSinkActor.remote()

        ref1 = source.make_tensor.remote()
        got1 = ray.get(sink.recv_and_verify.remote(ref1))
        src_state_after_first = ray.get(source.get_cache_state.remote())
        dst_state_after_first = ray.get(sink.get_cache_state.remote())

        ref2 = source.make_tensor.remote()
        got2 = ray.get(sink.recv_and_verify.remote(ref2))
        src_state_after_second = ray.get(source.get_cache_state.remote())
        dst_state_after_second = ray.get(sink.get_cache_state.remote())

        expected = list(range(12))
        assert got1 == expected, f"first transfer wrong: {got1}"
        assert got2 == expected, f"second transfer wrong: {got2}"

        assert src_state_after_first["hixl_initialized"] is True
        assert src_state_after_second["hixl_initialized"] is True

        assert src_state_after_first["tensor_desc_cache_size"] == 1
        assert src_state_after_second["tensor_desc_cache_size"] == 2

        assert dst_state_after_first["remote_engines_size"] == 1
        assert dst_state_after_second["remote_engines_size"] == 1
