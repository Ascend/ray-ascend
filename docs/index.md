# Ray-Ascend Documentation

Welcome to the official documentation for **Ray-Ascend** — a high-performance extension for [Ray](https://github.com/ray-project/ray) that enables efficient tensor transport and collective communication on Huawei Ascend AI accelerators.

## Overview

Ray-Ascend provides two core capabilities:

- **Direct Tensor Transport**: Zero-copy, low-latency tensor transfer between Ray actors/workers using Ascend-native mechanisms.
- **HCCL-based Collective Communication**: Scalable all-reduce, broadcast, and other collective operations via Huawei's HCCL backend.

This documentation is divided into two parts:

- **[User Guide](user_guide/index.md)**: For end users who want to install and use Ray-Ascend in their applications.
- **[Developer Guide](developer_guide/index.md)**: For contributors who want to understand, extend, or debug the codebase.

>  **Note**: This project assumes you have a working Ascend (e.g., Atlas 900) environment with CANN installed.
