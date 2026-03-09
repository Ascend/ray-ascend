# Contributing

> _Last upated: 03/09/2026_

## Instructions for Contribution

### Install pre-commit

`pre-commit` can automatically run various checks and fix tools before code is
committed, ensuring code quality and consistency.

```bash
# cd ray-ascend and launch pre-commit
pip install pre-commit
pre-commit install

# set local configuration
git config user.name "Your Name"
git config user.email "your.email@example.com"

# commit with signature, and then pre-commit would be triggered
git commit -s
```

For details about code style, please refer to the following:

#### Code style

Class naming style uses upper camelCase, for examples:

```python
class HCCLRootInfoStore:
    ...
class YRTensorTransport:
    ...
```

Local variables and methods use snake_case, for examples:

```python
def get_communicator_metadata():
    ...

class MyCollectiveGroup:
    def send_tensor(self):
        ...
```

Global variables and environment variables use upper snake_case, for examples:

```python
YR_DS_WORKER_HOST
YR_DS_WORKER_PORT
```

When you define a Python method, please add a type annotation like:

```python
def extract_tensor_transport_metadata(
        self,
        obj_id: str,
        gpu_object: List["torch.Tensor"],
    ) -> YRTransportMetadata:
    ...
```

## Run tests

All tests programs are located in the `tests/` directory and are written based on the
pytest framework.

```bash
# run tests
pip install -e ".[all]"
pytest -v
```

## Sign Ascend CLA

When you submit your PR for the first time, please sign the Ascend
[CLA ( Contributor License Agreement )](https://clasign.osinfra.cn/sign/690ca9ddf91c03dee6082ab1).
The email address used to sign the CLA must match the commit signature.
