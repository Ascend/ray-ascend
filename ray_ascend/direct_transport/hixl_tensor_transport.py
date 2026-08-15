import logging
import pickle
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, cast

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

try:
    import hixl
except ImportError:
    hixl = None

# Maximum number of cached HIXL remote engine connections.
# When exceeded, the least recently used remote engine is evicted and
# Disconnect is called. Set to 0 to disable remote engine reuse.
HIXL_REMOTE_ENGINE_CACHE_MAXSIZE = 1000


@dataclass
class HixlCommunicatorMetadata(CommunicatorMetadata):
    """Metadata for the HIXL communicator."""


@dataclass
class HixlTransportMetadata(TensorTransportMetadata):
    """Metadata for tensors stored in the NPU/CPU object store for HIXL transport.

    Args:
        hixl_serialized_mem_descs: Pickle-serialized list of
            (data_ptr, nbytes, mem_type_str) tuples describing the source
            tensors' registered memory regions.
        hixl_engine_id: The local HIXL engine identifier (format: "host_ip:port")
            that the remote side uses to Connect back.
        hixl_mem_generation: Monotonically increasing memory-registration
            generation number bumped whenever a registered memory region is
            deregistered, so the receiver can detect stale descriptors.
    """

    hixl_serialized_mem_descs: Optional[bytes] = None
    hixl_engine_id: Optional[str] = None
    hixl_mem_generation: Optional[int] = 0

    __eq__ = object.__eq__
    __hash__ = object.__hash__


@dataclass
class HixlTensorDesc:
    """Cached registration info for a single tensor storage.

    HIXL's RegisterMem returns only a MemHandle (void*), which does not carry
    address or size information. We keep the original registration parameters
    alongside the handle so we can:
      - Build TransferOpDesc tuples on the source side (addr, len are needed)
      - Call DeregisterMem(mem_handle) when the ref count drops to zero
      - Serialize (data_ptr, nbytes, mem_type_str) into transport metadata

    Attributes:
        mem_handle: The opaque handle returned by engine.register_mem.
            Represented as a Python int (uintptr_t under the hood).
        nbytes: Size of the registered memory region in bytes.
        mem_type_str: "npu" or "cpu" — used when building TransferOpDesc and
            for serialization into HixlTransportMetadata.
        metadata_count: Number of HixlTransportMetadata objects that reference
            this tensor. When it reaches zero, we call DeregisterMem.
    """

    mem_handle: int
    nbytes: int
    mem_type_str: str
    metadata_count: int


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
            Set to None by a cleanup path that has taken responsibility for
            cleanup, so __del__ won't re-enter _cleanup_transfer.
    """

    transfer_req: Optional[int] = None
    remote_engine_id: Optional[str] = None
    remove_tensor_descs: bool = False
    transport: Optional["HixlTensorTransport"] = None

    def __del__(self):
        if self.transport is not None:
            self.transport._cleanup_transfer(
                self.obj_id,
                self.tensors,
                self.transfer_req,
                self.remote_engine_id,
                self.remove_tensor_descs,
            )


class HixlTensorTransport(TensorTransportManager):
    """HIXL Engine-based one-sided RDMA tensor transport for Ray RDT."""

    def __init__(self):
        # Lazily initialized because hixl may not be installed on
        # nodes that are only coordinating (not participating in transfers).
        self._hixl_initialized = False
        self._local_engine_id: Optional[str] = None
        self._hixl_engine: Optional["hixl.Hixl"] = None

        # Object IDs whose transfers have been aborted.
        self._aborted_transfer_obj_ids: Set[str] = set()
        self._aborted_transfer_obj_ids_lock = threading.Lock()

        # Mapping from tensor storage data_ptr → HixlTensorDesc.
        self._tensor_desc_cache: Dict[int, HixlTensorDesc] = {}

        # Mapping from Ray object ID → HixlTransportMetadata.
        # Lifetime is tied to the object ref; freed when the ref goes out of
        # scope (garbage_collect is called).
        self._managed_meta_hixl: Dict[str, HixlTransportMetadata] = {}

        # Lock protecting _tensor_desc_cache, _managed_meta_hixl and
        # _hixl_mem_generation since they can be accessed from the main task
        # execution thread or the _ray_system thread.
        self._cache_lock = threading.RLock()

        # LRU cache of remote engine connections.
        # Key:   str  — remote engine id ("host_ip:port") that this engine
        #              has connected to.
        # Value: int  — the remote engine's mem generation.
        # When full, the least recently used remote engine is evicted and
        # Disconnect is called.
        self._remote_engines: "OrderedDict[str, int]" = OrderedDict()

        # Memory deregistration generation: incremented whenever a registered
        # memory region is deregistered, so receivers can detect stale
        # descriptors.
        self._hixl_mem_generation: int = 0

    def tensor_transport_backend(self) -> str:
        return "HIXL"

    @staticmethod
    def is_one_sided() -> bool:
        return True

    @staticmethod
    def can_abort_transport() -> bool:
        return True

    def finalize(self) -> None:
        """Explicitly release the HIXL engine's process-level resources."""
        if not self._hixl_initialized:
            return
        try:
            assert self._hixl_engine is not None
            self._hixl_engine.finalize()
        except Exception:
            logger.warning("HIXL engine finalize raised an exception", exc_info=True)
        finally:
            self._hixl_initialized = False
            self._hixl_engine = None
            self._remote_engines.clear()

    @staticmethod
    def _allocate_listen_port() -> int:
        """Reserve a free TCP port for the HIXL engine to listen on."""
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("", 0))
            return cast(int, sock.getsockname()[1])
        finally:
            sock.close()

    @staticmethod
    def _resolve_npu_device_id() -> int:
        """Return the logical NPU index this process should bind hixl to."""
        return 0

    def _ensure_hixl_initialized(self) -> None:
        """Lazily initialize the HIXL engine.

        Raises:
            ImportError: If hixl is not installed.
            RuntimeError: If HIXL construction or initialize fails.
        """
        if self._hixl_initialized:
            return

        if hixl is None:
            raise ImportError("hixl module not found. ")

        with self._cache_lock:
            if self._hixl_initialized:
                return

            node_ip = ray.util.get_node_ip_address()
            listen_port = self._allocate_listen_port()
            self._local_engine_id = f"{node_ip}:{listen_port}"
            import torch

            torch.npu.set_device(self._resolve_npu_device_id())

            try:
                self._hixl_engine = hixl.Hixl()
                options = {hixl.OPTION_AUTO_CONNECT: "1"}
                status = self._hixl_engine.initialize(self._local_engine_id, options)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize HIXL engine with id "
                    f"'{self._local_engine_id}'. "
                    f"Original error: {e}"
                ) from e

            if status != hixl.SUCCESS:
                raise RuntimeError(
                    f"HIXL initialize returned error status={status} "
                    f"for engine id '{self._local_engine_id}'. "
                )

            self._hixl_initialized = True
            logger.info(
                f"HIXL engine initialized with "
                f"local_engine_id={self._local_engine_id}"
            )

    def actor_has_tensor_transport(self, actor: "ray.actor.ActorHandle") -> bool:
        """Check if a remote actor has the HIXL transport available."""

        # TODO: This is called on a .remote RDT call, so it's quite expensive.
        def __ray_actor_has_tensor_transport__(
            self: "ray.actor.ActorHandle",
        ) -> bool:
            try:
                from ray.experimental.rdt.util import get_tensor_transport_manager

                manager = get_tensor_transport_manager("HIXL")
                manager._ensure_hixl_initialized()
                return True
            except Exception:
                return False

        result = ray.get(
            actor.__ray_call__.options(concurrency_group="_ray_system").remote(
                __ray_actor_has_tensor_transport__
            )
        )
        return bool(result)

    def register_hixl_memory(self, tensor: "torch.Tensor") -> None:
        """Registers the tensor's memory with HIXL and bumps the reference
        count so the memory region is not deregistered for the lifetime of the
        process.
        """
        self._add_tensor_descs([tensor])

    def deregister_hixl_memory(self, tensor: "torch.Tensor") -> None:
        """Decrements the reference count for the tensor's HIXL memory
        registration added by register_hixl_memory.

        If the reference count reaches 0, the memory is deregistered from
        HIXL. This should only be called after register_hixl_memory has been
        called for this tensor. Any existing ObjectRef instances that reference
        this tensor's memory will keep the HIXL registration alive independently
        until they go out of scope.
        """
        self._remove_tensor_descs([tensor])

    def _add_tensor_descs(self, tensors: List["torch.Tensor"]):
        """Register tensor memory with HIXL and bump reference counts.

        If a tensor's storage is already registered, we
        only increment the metadata_count. Otherwise we call
        engine.register_mem and cache the handle + registration params.
        """
        self._ensure_hixl_initialized()
        assert self._hixl_engine is not None

        with self._cache_lock:
            for tensor in tensors:
                key = tensor.untyped_storage().data_ptr()
                if key in self._tensor_desc_cache:
                    self._tensor_desc_cache[key].metadata_count += 1
                    continue

                mem_type_str = "npu" if tensor.device.type == "npu" else "cpu"

                addr = tensor.untyped_storage().data_ptr()
                nbytes = tensor.untyped_storage().nbytes()
                mem_type = (
                    hixl.MemType.MEM_DEVICE
                    if mem_type_str == "npu"
                    else hixl.MemType.MEM_HOST
                )

                try:
                    status, mem_handle = self._hixl_engine.register_mem(
                        hixl.MemDesc(addr, nbytes), mem_type
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to register {mem_type_str} memory with HIXL "
                        f"(addr=0x{addr:x}, size={nbytes} bytes). "
                    ) from e

                if status != hixl.SUCCESS:
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

    def _remove_tensor_descs(self, tensors: List["torch.Tensor"]):
        """Decrement reference counts and deregister when they reach zero."""
        with self._cache_lock:
            for tensor in tensors:
                key = tensor.untyped_storage().data_ptr()
                if key not in self._tensor_desc_cache:
                    continue
                tensor_desc = self._tensor_desc_cache[key]
                tensor_desc.metadata_count -= 1
                if tensor_desc.metadata_count == 0:
                    self._tensor_desc_cache.pop(key)
                    assert self._hixl_engine is not None
                    try:
                        status = self._hixl_engine.deregister_mem(
                            tensor_desc.mem_handle
                        )
                        if status != hixl.SUCCESS:
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
                    self._hixl_mem_generation += 1

    def _tensor_memory_registered(self, t: "torch.Tensor") -> bool:
        """Check if the tensor's memory has been registered with HIXL."""
        return t.untyped_storage().data_ptr() in self._tensor_desc_cache

    def extract_tensor_transport_metadata(
        self,
        obj_id: str,
        rdt_object: List["torch.Tensor"],
    ) -> HixlTransportMetadata:
        """Source side: register tensor memory and serialize descriptors.

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
                device = rdt_object[0].device
                devices = set()
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
                    for dev in devices:
                        torch.npu.synchronize(dev)

                self._add_tensor_descs(rdt_object)

                for t in rdt_object:
                    key = t.untyped_storage().data_ptr()
                    desc = self._tensor_desc_cache[key]
                    mem_descs_for_serialization.append(
                        (key, desc.nbytes, desc.mem_type_str)
                    )

                serialized_mem_descs = pickle.dumps(mem_descs_for_serialization)
                engine_id = self._local_engine_id
                engine_mem_generation = self._hixl_mem_generation
            else:
                serialized_mem_descs = None
                engine_id = None
                engine_mem_generation = None

            ret = HixlTransportMetadata(
                tensor_meta=tensor_meta,
                tensor_device=device.type if device else None,
                hixl_serialized_mem_descs=serialized_mem_descs,
                hixl_engine_id=engine_id,
                hixl_mem_generation=engine_mem_generation,
            )
            self._put_meta(obj_id, ret)
            return ret

    def get_communicator_metadata(
        self,
        src_actor: "ray.actor.ActorHandle",
        dst_actor: "ray.actor.ActorHandle",
        backend: Optional[str] = None,
    ) -> HixlCommunicatorMetadata:
        """One-sided RDMA transport: no communicator metadata needed."""
        return HixlCommunicatorMetadata()

    def fetch_multiple_tensors(
        self,
        obj_id: str,
        tensor_transport_metadata: HixlTransportMetadata,
        communicator_metadata: HixlCommunicatorMetadata,
        target_buffers: Optional[List["torch.Tensor"]] = None,
    ) -> HixlFetchRequest:
        """Receiver side: initiate an RDMA READ transfer.

        This triggers the transfer but does not wait for completion. Call
        wait_fetch_complete(fetch_request) to retrieve the tensors.

        Args:
            obj_id: The object ID for the transfer.
            tensor_transport_metadata: Source-side metadata containing
                serialized memory descriptions and the remote engine ID.
            communicator_metadata: Empty HixlCommunicatorMetadata.
            target_buffers: Optional pre-allocated buffers to receive into.

        Returns:
            HixlFetchRequest carrying the async transfer state.
        """
        from ray.experimental.rdt.util import create_empty_tensors_from_metadata

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
                raise RuntimeError(f"HIXL transfer aborted for object id: {obj_id}")

        transfer_req = None
        added_tensor_descs = False
        fetch_request = None

        assert tensors

        try:
            self._ensure_hixl_initialized()
            assert self._hixl_engine is not None
            self._add_tensor_descs(tensors)
            added_tensor_descs = True

            assert serialized_mem_descs is not None
            remote_mem_descs = pickle.loads(serialized_mem_descs)

            remote_engine_mem_generation = tensor_transport_metadata.hixl_mem_generation
            assert remote_engine_id is not None
            assert remote_engine_mem_generation is not None
            self._connect_remote_engine(remote_engine_id, remote_engine_mem_generation)

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
                op_descs.append(
                    hixl.TransferOpDesc(local_addr, remote_addr, remote_nbytes)
                )

            status, transfer_req = self._hixl_engine.transfer_async(
                remote_engine_id, hixl.TransferOp.READ, op_descs
            )

            if status != hixl.SUCCESS:
                raise RuntimeError(
                    f"HIXL TransferAsync returned error status={status} "
                    f"for object id: {obj_id}"
                )

            fetch_request = HixlFetchRequest(
                obj_id=obj_id,
                tensors=tensors,
                transfer_req=transfer_req,
                remote_engine_id=remote_engine_id,
                remove_tensor_descs=added_tensor_descs,
                transport=self,
            )
            return fetch_request
        except Exception:
            self._cleanup_transfer(
                obj_id,
                tensors,
                transfer_req,
                remote_engine_id,
                added_tensor_descs,
            )
            if fetch_request is not None:
                fetch_request.transport = None

            from ray.exceptions import RayDirectTransportError

            raise RayDirectTransportError(
                f"The HIXL transfer failed for object id: {obj_id}. "
                f"The source actor may have died during the transfer. "
                f"The exception thrown from HIXL transfer was:\n "
                f"{traceback.format_exc()}"
            ) from None

    def wait_fetch_complete(
        self, fetch_request: HixlFetchRequest, timeout: float = -1
    ) -> List["torch.Tensor"]:
        """Wait for a previously initiated HIXL fetch to complete.

        Polls engine.get_transfer_status until the state is "COMPLETED",
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
            return cast(List["torch.Tensor"], fetch_request.tensors)

        try:
            assert self._hixl_engine is not None
            deadline = None if timeout < 0 else time.monotonic() + timeout
            while True:
                status, transfer_status = self._hixl_engine.get_transfer_status(
                    fetch_request.transfer_req
                )
                if status != hixl.SUCCESS:
                    raise RuntimeError(
                        f"HIXL GetTransferStatus returned error status={status} "
                        f"for object id: {obj_id}"
                    )

                if transfer_status == hixl.TransferStatus.FAILED:
                    raise RuntimeError(
                        f"HIXL transfer got FAILED state for object id: {obj_id}"
                    )
                if transfer_status == hixl.TransferStatus.TIMEOUT:
                    raise RuntimeError(
                        f"HIXL transfer got TIMEOUT state for object id: {obj_id}"
                    )
                if transfer_status == hixl.TransferStatus.WAITING:
                    with self._aborted_transfer_obj_ids_lock:
                        if obj_id in self._aborted_transfer_obj_ids:
                            self._aborted_transfer_obj_ids.remove(obj_id)
                            raise RuntimeError(
                                f"HIXL transfer aborted for object id: {obj_id}"
                            )
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"HIXL transfer timed out after {timeout}s "
                            f"for object id: {obj_id}"
                        )
                    time.sleep(0.001)  # Avoid busy waiting
                elif transfer_status == hixl.TransferStatus.COMPLETED:
                    break

            return cast(List["torch.Tensor"], fetch_request.tensors)
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

    def _cleanup_transfer(
        self,
        obj_id: str,
        tensors: List["torch.Tensor"],
        transfer_req: Optional[int],
        remote_engine_id: Optional[str],
        remove_tensor_descs: bool,
    ) -> None:
        """Best-effort cleanup after a transfer completes or fails."""
        if not self._hixl_initialized:
            return

        with self._aborted_transfer_obj_ids_lock:
            self._aborted_transfer_obj_ids.discard(obj_id)

        if HIXL_REMOTE_ENGINE_CACHE_MAXSIZE == 0 and remote_engine_id:
            self._disconnect_remote_engine(remote_engine_id)

        if remove_tensor_descs:
            self._remove_tensor_descs(tensors)

    def _connect_remote_engine(
        self, remote_engine_id: str, remote_engine_mem_generation: int
    ) -> None:
        """Connect to a remote HIXL engine, with LRU caching.

        Behavior depends on cache state and the remote memory-generation
        version (which changes when the source deregisters memory):

          - Case 1: already cached + version match -> reuse the connection
            (move to end of LRU), return without re-connecting.
          - Case 2: already cached + version mismatch -> disconnect, then
            reconnect and update the cached version.
          - Case 3: not cached + cache not full -> connect and cache it.
          - Case 4: not cached + cache full -> evict the least recently used
            engine, then connect and cache the new one.

        When the LRU cache is disabled (HIXL_REMOTE_ENGINE_CACHE_MAXSIZE == 0),
        the cache is never consulted: every call connects directly without
        caching.
        """
        with self._cache_lock:
            assert self._hixl_engine is not None
            if HIXL_REMOTE_ENGINE_CACHE_MAXSIZE > 0:
                if remote_engine_id in self._remote_engines:
                    cached_version = self._remote_engines[remote_engine_id]
                    if cached_version != remote_engine_mem_generation:
                        self._disconnect_remote_engine(remote_engine_id)
                    else:
                        self._remote_engines.move_to_end(remote_engine_id)
                        return

                elif len(self._remote_engines) >= HIXL_REMOTE_ENGINE_CACHE_MAXSIZE:
                    evicted_engine_id, _ = self._remote_engines.popitem(last=False)
                    self._disconnect_remote_engine(evicted_engine_id)

                status = self._hixl_engine.connect(remote_engine_id)
                if status != hixl.SUCCESS and status != hixl.ALREADY_CONNECTED:
                    raise RuntimeError(
                        f"HIXL Connect to '{remote_engine_id}' failed, "
                        f"status={status}"
                    )

                self._remote_engines[remote_engine_id] = remote_engine_mem_generation
            else:
                status = self._hixl_engine.connect(remote_engine_id)
                if status != hixl.SUCCESS and status != hixl.ALREADY_CONNECTED:
                    raise RuntimeError(
                        f"HIXL Connect to '{remote_engine_id}' failed, "
                        f"status={status}"
                    )

    def _disconnect_remote_engine(self, remote_engine_id: str) -> None:
        """Disconnect from a remote HIXL engine (best-effort)."""
        assert self._hixl_engine is not None
        try:
            self._hixl_engine.disconnect(remote_engine_id)
        except Exception:
            logger.warning(
                f"HIXL Disconnect from '{remote_engine_id}' raised exception",
                exc_info=True,
            )

    def recv_multiple_tensors(
        self,
        obj_id: str,
        tensor_transport_metadata: HixlTransportMetadata,
        communicator_metadata: HixlCommunicatorMetadata,
        target_buffers: Optional[List["torch.Tensor"]] = None,
    ) -> List["torch.Tensor"]:
        """Receives multiple tensors synchronously (fetch + wait)."""
        fetch_request = self.fetch_multiple_tensors(
            obj_id,
            tensor_transport_metadata,
            communicator_metadata,
            target_buffers,
        )
        return self.wait_fetch_complete(fetch_request)

    def send_multiple_tensors(
        self,
        tensors: List["torch.Tensor"],
        tensor_transport_metadata: HixlTransportMetadata,
        communicator_metadata: HixlCommunicatorMetadata,
    ):
        """Not implemented — HIXL is a one-sided transport."""
        raise NotImplementedError(
            "HIXL transport does not support send_multiple_tensors, "
            "since it is a one-sided transport."
        )

    def garbage_collect(
        self,
        obj_id: str,
        tensor_transport_meta: HixlTransportMetadata,
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

    def abort_transport(
        self,
        obj_id: str,
        communicator_metadata: HixlCommunicatorMetadata,
    ):
        """Mark a transfer as aborted so wait_fetch_complete can exit."""
        with self._aborted_transfer_obj_ids_lock:
            self._aborted_transfer_obj_ids.add(obj_id)

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
