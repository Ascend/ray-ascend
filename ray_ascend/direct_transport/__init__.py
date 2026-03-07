import warnings

from .hccl_tensor_transport import (
    HCCLTensorTransport,
)

__all__ = [
    "HCCLTensorTransport",
]

try:
    from .yr_tensor_transport import (
        YRTensorTransport,
    )
except ImportError:
    warnings.warn(
        "YRTensorTransport is not available because the optional 'yr' "
        "dependency is not installed.",
        ImportWarning,
    )
else:
    __all__.append("YRTensorTransport")
