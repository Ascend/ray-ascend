import pickle
from typing import Sequence

import torch

bytestr = bytes | bytearray | memoryview


class SimpleTensorEncoder:
    """A minimal encoder that only handles a single torch.Tensor.

    It mimics the interface of MsgpackEncoder.encode() but skips msgpack entirely.
    """

    def encode(self, obj: torch.Tensor) -> Sequence[bytestr]:
        """
        Encode a single torch.Tensor in zero-copy mode.

        Returns:
            A list [meta_bytes, raw_data_buffer] which is compatible with
            the original MsgpackEncoder's return type.
        """
        if not isinstance(obj, torch.Tensor):
            raise TypeError("SimpleTensorEncoder only supports torch.Tensor")

        if not obj.is_contiguous():
            obj = obj.contiguous()

        if obj.is_sparse or obj.is_nested:
            raise ValueError("Only regular dense tensors are supported.")

        # Extract raw data
        arr = obj.flatten().view(torch.uint8).numpy()
        raw_data = memoryview(arr)

        # Build metadata tuple (dtype, shape)
        dtype_str = str(obj.dtype).removeprefix("torch.")
        meta_tuple = (dtype_str, tuple(obj.shape))

        # Serialize metadata to bytes using pickle (for simplicity and compatibility)
        meta_bytes = pickle.dumps(meta_tuple, protocol=pickle.HIGHEST_PROTOCOL)

        bufs: list[bytestr] = [meta_bytes, raw_data]
        return bufs


class SimpleTensorDecoder:
    """A minimal decoder that only handles a single torch.Tensor encoded by SimpleTensorEncoder.

    It mimics the interface of MsgpackDecoder.decode() but skips msgpack entirely.
    """

    def decode(self, bufs: Sequence[bytestr]) -> torch.Tensor:
        """
        Decode a list of bytes into a torch.Tensor.

        Args:
            bufs: A sequence [meta_bytes, raw_data_buffer].

        Returns:
            The reconstructed torch.Tensor.
        """
        if isinstance(bufs, bytestr):
            raise ValueError(
                "SimpleTensorDecoder expects a sequence of buffers, not a single bytes object."
            )

        assert len(bufs) >= 2, "Expected at least [meta, data]"
        meta_bytes = bufs[0]
        dtype_str, shape = pickle.loads(meta_bytes)

        buffer = bufs[1]
        torch_dtype = getattr(torch, dtype_str)

        if not buffer:  # Handle empty tensors
            return torch.empty(shape, dtype=torch_dtype)
        # Zero-copy reconstruction
        # Create uint8 tensor from buffer, then view as original dtype and reshape
        arr = torch.frombuffer(buffer, dtype=torch.uint8)
        tensor = arr.view(torch_dtype).view(shape)
        return tensor


_encoder = SimpleTensorEncoder()
_decoder = SimpleTensorDecoder()
