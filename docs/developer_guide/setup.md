# Setup

We provide installation instructions for YuanRong direct transport and HCCL collective
group, and you can selectively install the relevant dependencies as needed.

### Install CANN

If you have NPU devices and want to accelerate the transmission of NPU tensor by **YR**
or **HCCL**, you need to install **Ascend-cann-toolkit**.

> **CANN** (Compute Architecture for Neural Networks) is a heterogeneous computing
> architecture launched by Huawei for AI scenarios. \
> HCCL(Huawei Collective Communication Library) is included in CANN.

We recommend developers to develop inside CANN container like:

```bash
# pull CANN container
docker pull swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.2.rc1-a3-ubuntu22.04-py3.11
```

To run a container based on this image, please refer to
[offical cann image documentations](https://github.com/Ascend/cann-container-image).

After CANN installation, confirm the toolkit path exists:

```bash
ls /usr/local/Ascend/ascend-toolkit/latest
```

Set environment:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## Install ray-ascend without yr

Clone the ray-ascend repository, then install it either from source or by building a
wheel package.

```bash
git clone https://github.com/Ascend/ray-ascend.git
cd ray-ascend
```

### Install from source codes

```bash
pip install -e .
```

### build a wheel package

```bash
pip install -r requirements.txt
pip install build
python -m build --wheel
# install wheel
pip install dist/*.whl
```

## Install ray-ascend with yr

If you want to use yr direct tensor transport, please install dependencies following
these steps.

### Install datasystem package

```bash
# Automatically install all python dependencies of yr
pip install -e ".[yr]"
```

Verify installation by checking for the `dscli` command-line tool.

```bash
#If the installation is successful, a string similar to "dscli 9.9.9" will be printed.
dscli -version
```

### Install etcd

Openyuanrong-datasystem relies on etcd for cluster coordination. Download and install
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
