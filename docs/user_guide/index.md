# User Guide

> _Last updated: 03/24/2026_

## Overview

ray-ascend is a community-maintained hardware plugin that supports advanced
[Ray](https://github.com/ray-project/ray) features on Ascend NPU accelerators.

This guide provides step-by-step instructions for installation, configuration, and usage
of ray-ascend's key features:

- **HCCL Collective Communication**: Distributed collective operations across Ray actors
  using Huawei Collective Communication Library
- **YR Direct Transport**: Efficient zero-copy transfer of CPU and NPU tensors between
  Ray actors

## Prerequisites

- **Architecture**: aarch64, x86
- **OS Kernel**: Linux
- **Python**: >= 3.10, \<= 3.11
- **Ray**: Same version as ray-ascend

Optional dependencies for specific features:

- **CANN == 8.2.rc1**: Required for NPU features (HCCL, NPU tensor transport)
- **torch == 2.7.1, torch-npu == 2.7.1.post1**: Required for PyTorch NPU support

## Quick Start

```bash
# Install with YR support
pip install "ray-ascend[yr]"
```

## Contents

- [Installation](installation.md): Detailed installation and setup instructions
- [HCCL Collective Communication](hccl_collective.md): Collective operations guide
- [YR Direct Transport](yr_transport.md): Tensor transport guide
- [API Reference](api_reference.md): Complete API documentation
- [Best Practices](best_practices.md): Best practices, troubleshooting, and FAQ

## Additional Resources

- [Ray Documentation](https://docs.ray.io/)
- [Ascend Documentation](https://www.hiascend.com/)
- [OpenYuanrong DataSystem Documentation](https://pages.openeuler.openatom.cn/openyuanrong-datasystem/docs/zh-cn/latest/)
- [GitHub Repository](https://github.com/Ascend/ray-ascend)
- [Developer Guide](../developer_guide/index.md)
