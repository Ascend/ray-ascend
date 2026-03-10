# Setup

> _Last updated: 03/09/2026_

We provide installation instructions for YuanRong direct transport and HCCL collective
communication, and you can selectively install the relevant dependencies as needed.

### Install CANN

If you have NPU devices and want to accelerate the transmission of NPU tensor by
**YuanRong** or **HCCL**, you need to install **Ascend CANN Toolkit**.

> **CANN** (Compute Architecture for Neural Networks) is a heterogeneous computing
> architecture launched by Huawei for AI scenarios.
>
> HCCL (Huawei Collective Communication Library) is included in CANN.

We recommend developing inside a CANN container.

First, please select the appropriate
[CANN image](https://hub.docker.com/r/ascendai/cann) (aligned with your **CANN
version**/**Ascend hardware**/**OS**/**Python version**) and pull it.

| CANN Version | Ascend Hardware | OS           | Python Version | Image Name                           |
| ------------ | --------------- | ------------ | -------------- | ------------------------------------ |
| 8.2.rc1      | A3              | Ubuntu 22.04 | 3.11           | cann:8.2.rc1-a3-ubuntu22.04-py3.11   |
| 8.2.rc1      | 910B            | Ubuntu 22.04 | 3.11           | cann:8.2.rc1-910b-ubuntu22.04-py3.11 |

```bash
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/${image name}

# for Ascend NPU A3
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.2.rc1-a3-ubuntu22.04-py3.11
# for Ascend NPU 910B
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.2.rc1-910b-ubuntu22.04-py3.11
```

To run a container based on this image, please refer to
[official CANN image documentation](https://github.com/Ascend/cann-container-image?tab=readme-ov-file#usage).

After CANN installation, confirm the toolkit path exists:

```bash
ls /usr/local/Ascend/ascend-toolkit/latest
```

## Install Minimum ray-ascend (HCCL Only)

Clone the ray-ascend repository, then install it either from source or by building a
wheel package.

```bash
git clone https://github.com/Ascend/ray-ascend.git
cd ray-ascend
```

### Install from Source

```bash
pip install -e .
```

### Build a Wheel Package

```bash
pip install -r requirements.txt
pip install build
python -m build --wheel
# Install the wheel
pip install dist/*.whl
```

## Install ray-ascend with YuanRong

If you want to use
[YR](https://pages.openeuler.openatom.cn/openyuanrong-datasystem/docs/zh-cn/latest/index.html)
direct tensor transport, please install the dependencies by following these steps.

### Install DataSystem Package

```bash
# Automatically install all Python dependencies for YR
pip install -e ".[yr]"
```

Verify the installation by checking for the `dscli` command-line tool.

```bash
# If the installation is successful, a string like "dscli 9.9.9" will be printed.
dscli --version
```

### Install etcd

OpenYuanRong DataSystem relies on etcd for cluster coordination. Download and install
etcd from the official releases:
[ETCD GitHub Releases](https://github.com/etcd-io/etcd/releases)

```bash
# Example for Linux ARM64 (adjust architecture as needed)
ETCD_VERSION="v3.6.5"  # Replace with your desired version
ARCH="linux-arm64"

# Unpack etcd
tar -xvf etcd-${ETCD_VERSION}-${ARCH}.tar.gz

# Create symbolic links in /usr/local/bin pointing directly into the extracted folder
sudo ln -sf "$(pwd)/etcd-${ETCD_VERSION}-${ARCH}/etcd" /usr/local/bin/etcd
sudo ln -sf "$(pwd)/etcd-${ETCD_VERSION}-${ARCH}/etcdctl" /usr/local/bin/etcdctl

# Verify installation
etcd --version
etcdctl version
```
