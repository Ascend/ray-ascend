# Installation

> _Last updated: 04/21/2026_

## Installation Options

### Basic Installation (HCCL Only)

Install the base package with HCCL collective communication support:

```bash
pip install ray-ascend
```

### With YR Direct Transport Support

Install with OpenYuanrong (YR) direct tensor transport support:

```bash
pip install "ray-ascend[yr]"
```

### From Source (Editable Installation)

For development or to use the latest version:

```bash
git clone https://github.com/Ascend/ray-ascend.git
cd ray-ascend
pip install -e ".[all]"
```

## CANN Setup (for NPU Features)

If you have Ascend NPU devices and want to use HCCL or NPU tensor transport, you need to
install the CANN toolkit.

### Using CANN Docker (Recommended)

We recommend using the official CANN Docker images for the easiest setup:

```bash
# For Ascend NPU A3
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.2.rc1-a3-ubuntu22.04-py3.11

# For Ascend NPU 910B
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.2.rc1-910b-ubuntu22.04-py3.11
```

For more details on running the container, see the
[official CANN image documentation](https://github.com/Ascend/cann-container-image).

### Verifying CANN Installation

After CANN installation, confirm the toolkit path exists:

```bash
ls /usr/local/Ascend/ascend-toolkit/latest
```

## Etcd Setup (Optional, for YR Transport etcd Mode)

OpenYuanrong DataSystem supports two initialization modes: `metastore` (default, no
external dependencies) and `etcd` (requires external etcd service). Etcd setup is only
needed if you choose to use etcd mode.

Download and install etcd from the official releases:
[ETCD GitHub Releases](https://github.com/etcd-io/etcd/releases)

```bash
# Example for Linux ARM64 (adjust architecture as needed)
ETCD_VERSION="v3.6.5"
ARCH="linux-arm64"  # or "linux-amd64" for x86

# Unpack etcd
tar -xvf etcd-${ETCD_VERSION}-${ARCH}.tar.gz

# Create symbolic links in /usr/local/bin
sudo ln -sf "$(pwd)/etcd-${ETCD_VERSION}-${ARCH}/etcd" /usr/local/bin/etcd
sudo ln -sf "$(pwd)/etcd-${ETCD_VERSION}-${ARCH}/etcdctl" /usr/local/bin/etcdctl

# Verify installation
etcd --version
etcdctl version
```

## Environment Variables for YR Transport

For YR transport environment variables configuration, see
[Environment Variables](yr_transport.md#environment-variables) in the YR Transport
guide.

Verify the YR installation by checking for the `dscli` command-line tool:

```bash
# If the installation is successful, a string like "dscli 9.9.9" will be printed.
dscli --version
```
