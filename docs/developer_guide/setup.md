# Setup
> last upate: 02/14/2026

We provide installation instructions for yuanrong direct transport and hccl collective group, and you can selectively install the relevant dependencies as needed.

## Install ray-ascend
Clone the ray-ascend repository, then install it either from source or by building a wheel package.
```bash
git clone https://github.com/Ascend/ray-ascend.git
cd ray-ascend
```
### Install from source codes
```
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

## (Optional) Install yuanrong-datasystem
If you want to use yr direct tensor transport, please install dependencies following these steps.
### Install datasystem package
```bash
# Install the OpenYuanrong Datasystem package
pip install openyuanrong-datasystem

# Verify installation by checking for the dscli command-line tool
dscli -h
```
### Install etcd
Openyuanrong-datasystem relies on etcd for cluster coordination. 
Download and install etcd from the official releases: [ETCD GitHub Releases](https://github.com/etcd-io/etcd/releases)

```bash
# Example for Linux ARM64 (adjust for your architecture)
# Unpack and install etcd
ETCD_VERSION = "v3.6.5" # Replace with the desired version
tar -xvf etcd-${ETCD_VERSION}-linux-arm64.tar.gz
cd etcd-${ETCD_VERSION}-linux-arm64

# Copy the executable file to the system path
sudo cp etcd etcdctl /usr/local/bin/

# Verify installation
etcd --version
etcdctl version
```
### Install CANN
If you have NPU devices and want to accelerate the transmission of NPU tensor, 
you can install **Ascend-cann-toolkit**.

> **CANN** (Compute Architecture for Neural Networks) is a heterogeneous computing architecture launched by Huawei for AI scenarios.

Download the appropriate toolkit package from:
[Ascend CANN Downloads](https://www.hiascend.com/developer/download/community/result?cann=8.3.RC1&product=1&model=30).

Please select the appropriate version for your OS and architecture (e.g., Linux + AArch64).

Then install the toolkit:
```bash
# For example, download the aarch64 package, set the execution permission, and install it.
chmod +x Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run
./Ascend-cann-toolkit_8.3.RC1_linux-aarch64.run --install

# Dependencies of CANN Installation
pip install scipy psutil tornado decorator ml-dtypes absl-py
```

After installation, confirm the toolkit path exists:
```bash
# Root user
ls /usr/local/Ascend/ascend-toolkit/latest

# Non-root user
ls ${HOME}/Ascend/ascend-toolkit/latest
```

> note: please check if the versions of torch and torch-npu are consistent.

## (Optional) Install HCCL
...