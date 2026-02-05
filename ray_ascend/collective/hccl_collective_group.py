import ctypes
import datetime
import logging
import time
from typing import Any, List, Optional, Sequence, Tuple

import ray
import torch
from ray.util.collective.collective_group.base_collective_group import BaseGroup
from ray.util.collective.const import get_store_name
from ray.util.collective.types import (
    AllGatherOptions,
    AllReduceOptions,
    Backend,
    BarrierOptions,
    BroadcastOptions,
    RecvOptions,
    ReduceOp,
    ReduceOptions,
    ReduceScatterOptions,
    SendOptions,
)

logger = logging.getLogger(__name__)


class HcclRootInfo(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 4108)]


buffer_type = ctypes.c_void_p
npuStream_t = ctypes.c_void_p
hcclComm_t = ctypes.c_void_p


class HcclDataTypeEnum:
    HCCL_DATA_TYPE_INT8 = 0
    HCCL_DATA_TYPE_INT16 = 1
    HCCL_DATA_TYPE_INT32 = 2
    HCCL_DATA_TYPE_FP16 = 3
    HCCL_DATA_TYPE_FP32 = 4
    HCCL_DATA_TYPE_INT64 = 5
    HCCL_DATA_TYPE_UINT64 = 6
    HCCL_DATA_TYPE_UINT8 = 7
    HCCL_DATA_TYPE_UINT16 = 8
    HCCL_DATA_TYPE_UINT32 = 9
    HCCL_DATA_TYPE_FP64 = 10
    HCCL_DATA_TYPE_BFP16 = 11
    HCCL_DATA_TYPE_INT128 = 12

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.HCCL_DATA_TYPE_INT8
        if dtype == torch.int16:
            return cls.HCCL_DATA_TYPE_INT16
        if dtype == torch.int32:
            return cls.HCCL_DATA_TYPE_INT32
        if dtype == torch.float16:
            return cls.HCCL_DATA_TYPE_FP16
        if dtype == torch.float32:
            return cls.HCCL_DATA_TYPE_FP32
        if dtype == torch.int64:
            return cls.HCCL_DATA_TYPE_INT64
        if dtype == torch.uint8:
            return cls.HCCL_DATA_TYPE_UINT8
        if dtype == torch.float64:
            return cls.HCCL_DATA_TYPE_FP64
        if dtype == torch.bfloat16:
            return cls.HCCL_DATA_TYPE_BFP16
        raise ValueError(f"Unsupported dtype: {dtype}")


class HcclRedOpTypeEnum:
    HCCL_REDUCE_SUM = 0
    HCCL_REDUCE_PROD = 1
    HCCL_REDUCE_MAX = 2
    HCCL_REDUCE_MIN = 3

    @classmethod
    def from_ray(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.HCCL_REDUCE_SUM
        if op == ReduceOp.PRODUCT:
            return cls.HCCL_REDUCE_PROD
        if op == ReduceOp.MAX:
            return cls.HCCL_REDUCE_MAX
        if op == ReduceOp.MIN:
            return cls.HCCL_REDUCE_MIN
        raise ValueError(f"Unsupported op: {op}")


@ray.remote
class HCCLRootInfoStore:
    """HcclRootInfo Store as a named actors class.

    Args:
        name: the unique name for this named actor.

    Attributes:
        name: the unique name for this named actor.
        root_info: the HcclRootInfo held in this store.
    """

    def __init__(self, name: str):
        self.name = name
        self.root_info_bytes: Optional[bytes] = None

    def set_root_info_bytes(self, root_info_bytes: bytes) -> bytes:
        self.root_info_bytes = root_info_bytes
        return self.root_info_bytes

    def get_root_info_bytes(self) -> Optional[bytes]:
        if not self.root_info_bytes:
            logger.warning(
                "The HCCLRootInfo has not been set yet for store {}.".format(self.name)
            )
        return self.root_info_bytes


class HCCLGroup(BaseGroup):
    def __init__(self, world_size: int, rank: int, group_name: str) -> None:
        """Init an HCCL collective group."""
        super(HCCLGroup, self).__init__(world_size, rank, group_name)
        # Initialize single communicator/stream used for both collective and p2p ops.
        self._comm: Optional[hcclComm_t] = None
        self._stream: Optional[torch.npu.Stream] = None
        self._store_name = get_store_name(f"{self.group_name}@collective")
        self._device: Optional[int] = None
        self._barrier_tensor: Optional[torch.Tensor] = None

        # Single communicator/stream for both collective and p2p ops.
        self.libhccl = ctypes.CDLL("libhccl.so")
        self._init_collective_communicator()

    def destroy_group(self) -> None:
        """Destroy communicator resources and cleanup the rendezvous store."""
        if self._comm is not None:
            try:
                result = self.libhccl.HcclCommDestroy(self._comm)
                logger.debug(f"HcclCommDestroy result: {result}")
            except Exception as exc:
                logger.warning(f"Failed to destroy communicator: {exc}")
            self._comm = None
            self._stream = None

        if self.rank == 0:
            self._destroy_store(self._store_name)
        self._barrier_tensor = None
        super(HCCLGroup, self).destroy_group()

    @classmethod
    def backend(cls) -> Backend:
        """Return the backend type for this group."""
        return Backend.HCCL

    def broadcast(
        self,
        tensor: torch.Tensor,
        broadcast_options: BroadcastOptions = BroadcastOptions(),
    ) -> None:
        """Broadcast a tensor to all other NPUs following options.

        Args:
            tensor: the tensor to be broadcast or received.
            broadcast_options: broadcast options.

        Returns:
            None
        """
        root_rank = broadcast_options.root_rank

        def collective_fn(
            input_tensor: torch.Tensor, output_tensor: torch.Tensor, comm, stream
        ):
            # HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
            with torch.npu.device(input_tensor.device):
                current_stream = torch.npu.current_stream()
                stream.wait_stream(current_stream)
                exec_result = self.libhccl.HcclBroadcast(
                    buffer_type(input_tensor.data_ptr()),
                    input_tensor.numel(),
                    HcclDataTypeEnum.from_torch(input_tensor.dtype),
                    root_rank,
                    comm,
                    npuStream_t(stream.npu_stream),
                )
                event = torch.npu.Event()
                event.record(stream)
                current_stream.wait_event(event)
            logger.debug(f"HcclBroadcast execute result : {exec_result}")

        input_tensor = self._validate_tensor(tensor)
        output_tensor = input_tensor
        comm, stream = self._validate_collective_state()
        collective_fn(input_tensor, output_tensor, comm, stream)

    def allgather(
        self,
        tensor_list: List[torch.Tensor],
        tensor: torch.Tensor,
        allgather_options: AllGatherOptions = AllGatherOptions(),
    ) -> None:
        """Allgather a tensor across NPUs into a list of tensors.

        Args:
            tensor_list (List[Tensor]): output list for gathered tensors.
            tensor: the input tensor to allgather across the group.
            allgather_options: allgather options.

        Returns:
            None
        """

        def collective_fn(
            input_tensor: torch.Tensor, output_tensor: torch.Tensor, comm, stream
        ):
            with torch.npu.device(input_tensor.device):
                # HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
                current_stream = torch.npu.current_stream()
                stream.wait_stream(current_stream)
                exec_result = self.libhccl.HcclAllGather(
                    buffer_type(input_tensor.data_ptr()),
                    buffer_type(output_tensor.data_ptr()),
                    input_tensor.numel(),
                    HcclDataTypeEnum.from_torch(input_tensor.dtype),
                    comm,
                    npuStream_t(stream.npu_stream),
                )
                event = torch.npu.Event()
                event.record(stream)
                current_stream.wait_event(event)
            logger.debug(f"HcclAllGather execute result : {exec_result}")

        output_flattened = [_flatten_for_scatter_gather(tensor_list, copy=False)]

        input_tensor = self._validate_tensor(tensor)
        output_tensor = self._validate_tensor(output_flattened[0])
        comm, stream = self._validate_collective_state()
        collective_fn(input_tensor, output_tensor, comm, stream)

        for j, out_tensor in enumerate(tensor_list):
            out_tensor.copy_(output_flattened[0][j])

    def allreduce(
        self,
        tensor: torch.Tensor,
        allreduce_options: AllReduceOptions = AllReduceOptions(),
    ) -> None:
        """AllReduce a tensor across the collective group following options.

        Args:
            tensor: the tensor to be reduced. Must reside on one NPU.
            allreduce_options: allreduce options.

        Returns:
            None
        """

        def collective_fn(
            input_tensor: torch.Tensor, output_tensor: torch.Tensor, comm, stream
        ):
            with torch.npu.device(input_tensor.device):
                # HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
                current_stream = torch.npu.current_stream()
                stream.wait_stream(current_stream)
                exec_result = self.libhccl.HcclAllReduce(
                    buffer_type(input_tensor.data_ptr()),
                    buffer_type(output_tensor.data_ptr()),
                    input_tensor.numel(),
                    HcclDataTypeEnum.from_torch(input_tensor.dtype),
                    HcclRedOpTypeEnum.from_ray(allreduce_options.reduceOp),
                    comm,
                    npuStream_t(stream.npu_stream),
                )
                event = torch.npu.Event()
                event.record(stream)
                current_stream.wait_event(event)
            logger.debug(f"HcclAllReduce execute result : {exec_result}")

        input_tensor = self._validate_tensor(tensor)
        output_tensor = input_tensor
        comm, stream = self._validate_collective_state()
        collective_fn(input_tensor, output_tensor, comm, stream)

    def barrier(self, barrier_options: BarrierOptions = BarrierOptions()) -> None:
        """Blocks until all processes reach this barrier.

        Args:
            barrier_options: barrier options.

        Returns:
            None
        """
        device = self._device
        if device is None:
            raise RuntimeError("Collective communicator is not initialized.")
        # Lazily initialize and reuse the barrier tensor to avoid repeated allocations.
        if self._barrier_tensor is None or self._barrier_tensor.device.index != device:
            with torch.npu.device(device):
                self._barrier_tensor = torch.ones(1).npu()
        self.allreduce(self._barrier_tensor)

    def reduce(
        self, tensor: torch.Tensor, reduce_options: ReduceOptions = ReduceOptions()
    ) -> None:
        """Reduce a tensor to a destination NPU following options.

        Args:
            tensor: the tensor to be reduced. Must reside on one NPU.
            reduce_options: reduce options.

        Returns:
            None
        """
        root_rank = reduce_options.root_rank

        def collective_fn(
            input_tensor: torch.Tensor, output_tensor: torch.Tensor, comm, stream
        ):
            with torch.npu.device(input_tensor.device):
                # HcclResult HcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint32_t root, HcclComm comm, aclrtStream stream)
                current_stream = torch.npu.current_stream()
                stream.wait_stream(current_stream)
                exec_result = self.libhccl.HcclReduce(
                    buffer_type(input_tensor.data_ptr()),
                    buffer_type(output_tensor.data_ptr()),
                    input_tensor.numel(),
                    HcclDataTypeEnum.from_torch(input_tensor.dtype),
                    HcclRedOpTypeEnum.from_ray(reduce_options.reduceOp),
                    root_rank,
                    comm,
                    npuStream_t(stream.npu_stream),
                )
                event = torch.npu.Event()
                event.record(stream)
                current_stream.wait_event(event)
                logger.debug(f"HcclReduce execute result : {exec_result}")

        input_tensor = self._validate_tensor(tensor)
        output_tensor = input_tensor
        comm, stream = self._validate_collective_state()
        collective_fn(input_tensor, output_tensor, comm, stream)

    def reducescatter(
        self,
        tensor: torch.Tensor,
        tensor_list: List[torch.Tensor],
        reducescatter_options: ReduceScatterOptions = ReduceScatterOptions(),
    ) -> None:
        """Reduce then scatter a list of tensors across the group.

        Args:
            tensor: the output tensor to receive this rank's shard.
            tensor_list (List[Tensor]): the list of tensors to be reduced then scattered.
            reducescatter_options: reduce-scatter options.

        Returns:
            None
        """

        def collective_fn(
            input_tensor: torch.Tensor, output_tensor: torch.Tensor, comm, stream
        ):
            with torch.npu.device(input_tensor.device):
                # HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
                current_stream = torch.npu.current_stream()
                stream.wait_stream(current_stream)
                exec_result = self.libhccl.HcclReduceScatter(
                    buffer_type(input_tensor.data_ptr()),
                    buffer_type(output_tensor.data_ptr()),
                    output_tensor.numel(),
                    HcclDataTypeEnum.from_torch(input_tensor.dtype),
                    HcclRedOpTypeEnum.from_ray(reducescatter_options.reduceOp),
                    comm,
                    npuStream_t(stream.npu_stream),
                )
                event = torch.npu.Event()
                event.record(stream)
                current_stream.wait_event(event)
                logger.debug(f"HcclReduceScatter execute result : {exec_result}")

        input_flattened = [_flatten_for_scatter_gather(tensor_list, copy=False)]

        for j, in_tensor in enumerate(tensor_list):
            input_flattened[0][j].copy_(in_tensor)

        input_tensor = self._validate_tensor(input_flattened[0])
        output_tensor = self._validate_tensor(tensor)
        comm, stream = self._validate_collective_state()
        collective_fn(input_tensor, output_tensor, comm, stream)

    def send(
        self, tensor: torch.Tensor, send_options: SendOptions = SendOptions()
    ) -> None:
        """Send a tensor to a destination NPU in the group.

        Args:
            tensor: the tensor to send.
            send_options: send options.

        Returns:
            None
        """

        def p2p_fn(tensor: torch.Tensor, comm, stream, peer):
            with torch.npu.device(f"npu:{tensor.device.index}"):
                # HcclResult HcclSend(void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,HcclComm comm, aclrtStream stream)
                current_stream = torch.npu.current_stream()
                stream.wait_stream(current_stream)
                exec_result = self.libhccl.HcclSend(
                    buffer_type(tensor.data_ptr()),
                    tensor.numel(),
                    HcclDataTypeEnum.from_torch(tensor.dtype),
                    peer,
                    comm,
                    npuStream_t(stream.npu_stream),
                )
                event = torch.npu.Event()
                event.record(stream)
                current_stream.wait_event(event)
            logger.debug(f"HcclSend execute result : {exec_result}")

        tensor = self._validate_tensor(tensor)
        comm, stream = self._validate_collective_state()
        p2p_fn(tensor, comm, stream, send_options.dst_rank)

    def recv(
        self, tensor: torch.Tensor, recv_options: RecvOptions = RecvOptions()
    ) -> None:
        """Receive a tensor from a source NPU in the group.

        Args:
            tensor: the received tensor.
            recv_options: Receive options.

        Returns:
            None
        """

        def p2p_fn(tensor: torch.Tensor, comm, stream, peer):
            with torch.npu.device(f"npu:{tensor.device.index}"):
                # HcclResult HcclRecv(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,HcclComm comm, aclrtStream stream)
                current_stream = torch.npu.current_stream()
                stream.wait_stream(current_stream)
                exec_result = self.libhccl.HcclRecv(
                    buffer_type(tensor.data_ptr()),
                    tensor.numel(),
                    HcclDataTypeEnum.from_torch(tensor.dtype),
                    peer,
                    comm,
                    npuStream_t(stream.npu_stream),
                )
                event = torch.npu.Event()
                event.record(stream)
                current_stream.wait_event(event)
            logger.debug(f"HcclRecv execute result : {exec_result}")

        tensor = self._validate_tensor(tensor)
        comm, stream = self._validate_collective_state()
        p2p_fn(tensor, comm, stream, recv_options.src_rank)

    def _generate_hccl_root_info(self, store_name: str, dev: int = 0) -> HcclRootInfo:
        root_info = HcclRootInfo()
        # NPU need set device before HcclGetRootInfo
        with torch.npu.device(f"npu:{dev}"):
            exec_result = self.libhccl.HcclGetRootInfo(ctypes.byref(root_info))
        logger.debug(f"HcclGetRootInfo execute result : {exec_result}")

        store = HCCLRootInfoStore.options(  # type: ignore[attr-defined]
            name=store_name, lifetime="detached"
        ).remote(store_name)
        # TODO:
        # Maybe Ray support ctypes.Structure, we don't need convert to bytes here.
        ray.get([store.set_root_info_bytes.remote(bytes(root_info))])

        return root_info

    def _get_store_ref(self, store_name: str, timeout_s: int = 30) -> Any:
        """Get the reference of the named actor store.

        Args:
            store_name: the unique store key
            timeout_s: timeout in seconds.

        Return:
            store_ref: reference to store actor
        """
        if timeout_s <= 0:
            raise ValueError(
                "The 'timeout' argument must be positive. "
                "Got '{}'.".format(timeout_s)
            )
        store_ref = None
        timeout_delta = datetime.timedelta(seconds=timeout_s)
        elapsed = datetime.timedelta(seconds=0)
        start_time = datetime.datetime.now()
        while elapsed < timeout_delta:
            try:
                logger.debug("Trying to meet at the store '{}'".format(store_name))
                store_ref = ray.get_actor(store_name)
            except ValueError:
                logger.debug(
                    "Failed to meet at the store '{}'."
                    "Trying again...".format(store_name)
                )
                time.sleep(1)
                elapsed = datetime.datetime.now() - start_time
                continue
            logger.info("Successful rendezvous!")
            break
        if not store_ref:
            raise RuntimeError(
                "Unable to meet other processes "
                "at the rendezvous store. If you are using "
                "P2P communication, please check if tensors "
                "are put in the correct NPU. "
            )
        return store_ref

    @staticmethod
    def _destroy_store(store_name: str) -> None:
        store = ray.get_actor(store_name)
        # ray.get([store.__ray_terminate__.remote()])
        ray.kill(store)

    def _get_hccl_root_info(self, store_ref: Any, timeout_s: int = 30) -> HcclRootInfo:
        """Get the HcclRootInfo from the store through Ray.

        Args:
            store_ref: reference to the rendezvous store actor.
            timeout_s: timeout in seconds.

        Return:
            root_info: the HcclRootInfo if successful.
        """
        root_info_bytes = None
        timeout_delta = datetime.timedelta(seconds=timeout_s)
        elapsed = datetime.timedelta(seconds=0)
        start_time = datetime.datetime.now()
        while elapsed < timeout_delta:
            root_info_bytes = ray.get(store_ref.get_root_info_bytes.remote())
            if not root_info_bytes:
                time.sleep(1)
                elapsed = datetime.datetime.now() - start_time
                continue
            break
        if not root_info_bytes:
            raise RuntimeError("Unable to get the HcclRootInfo from the store.")
        return HcclRootInfo.from_buffer_copy(bytearray(root_info_bytes))

    def _init_collective_communicator(self) -> None:
        """Initialize the single communicator/stream for this group."""
        if self._comm is not None:
            return
        device = torch.npu.current_device()
        if self.rank == 0:
            root_info = self._generate_hccl_root_info(
                store_name=self._store_name, dev=device
            )
        else:
            store_ref = self._get_store_ref(store_name=self._store_name)
            root_info = self._get_hccl_root_info(store_ref)

        with torch.npu.device(f"npu:{device}"):
            comm: hcclComm_t = hcclComm_t()
            result = self.libhccl.HcclCommInitRootInfo(
                self.world_size,
                ctypes.byref(root_info),
                self.rank,
                ctypes.byref(comm),
            )
            logger.debug(
                f"Process {self.rank} Device {device}, return of `HcclCommInitRootInfo`: {result}"
            )
            stream = torch.npu.Stream()

        self._comm = comm
        self._stream = stream
        self._device = device

    def _validate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Validate a single tensor and return it.

        Enforces the single-device constraint and checks that the tensor device
        matches the communicator's initialization device.
        """
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError("Collective ops require torch.Tensor inputs.")
        device = get_tensor_device(tensor)
        if self._device is None:
            raise RuntimeError("Collective communicator is not initialized.")
        if device != self._device:
            raise RuntimeError(
                "Collective ops must use the same device as communicator initialization."
            )
        return tensor

    def _validate_collective_state(self) -> Tuple[hcclComm_t, torch.npu.Stream]:
        """Validate communicator and stream state and return them."""
        if self._comm is None or self._stream is None:
            raise RuntimeError("Collective communicator is not initialized.")
        return self._comm, self._stream


def get_tensor_device(tensor: torch.Tensor) -> int:
    """Return the NPU index of a tensor."""
    if isinstance(tensor, torch.Tensor):
        device = tensor.device.index
        if not isinstance(device, int):
            raise RuntimeError("The tensor is not on a valid NPU.")
    else:
        raise ValueError("Unsupported tensor type. Got: {}.".format(type(tensor)))
    return device


def _flatten_for_scatter_gather(
    tensor_list: Sequence[torch.Tensor], copy: bool = False
) -> torch.Tensor:
    """Flatten the tensor for gather/scatter operations.

    Args:
        tensor_list: the list of tensors to be scattered/gathered.
        copy: whether to copy the tensors in tensor_list into the buffer.

    Returns:
        The flattened tensor buffer.
    """
    if not tensor_list:
        raise RuntimeError("Received an empty list.")
    t: torch.Tensor = tensor_list[0]
    buffer_shape = [len(tensor_list)] + list(t.shape)

    buffer = torch.empty(tuple(buffer_shape), dtype=t.dtype, device=t.device)
    if copy:
        for i, tensor in enumerate(tensor_list):
            buffer[i].copy_(tensor)
    return buffer
