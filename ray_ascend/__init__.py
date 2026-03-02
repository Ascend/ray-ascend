"""
ray-ascend is a community maintained hardware plugin to support advanced Ray features
on Ascend NPU accelerators.
"""

from ray_ascend import _version

__all__ = [
    "__version__",
    "__commit__",
]

__commit__ = _version.commit
__version__ = _version.version
