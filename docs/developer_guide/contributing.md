# Contributing

> _Last updated: 03/09/2026_

## Contribution Guidelines

### Install Pre-Commit

`pre-commit` automatically runs various checks and fixes before code is committed,
ensuring code quality and consistency.

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

For detailed coding guidelines, please refer to the following:

#### Code style

Class names should use UpperCamelCase, as in these examples:

```python
class HCCLRootInfoStore:
    ...
class YRTensorTransport:
    ...
```

Local variables and methods should use snake_case, as shown here:

```python
def get_communicator_metadata():
    ...

class MyCollectiveGroup:
    def send_tensor(self):
        ...
```

Global variables and environment variables should use UPPER_SNAKE_CASE, as in these
examples:

```python
YR_DS_WORKER_HOST
YR_DS_WORKER_PORT
```

When defining a Python method, please add type annotations as follows:

```python
def extract_tensor_transport_metadata(
        self,
        obj_id: str,
        gpu_object: List["torch.Tensor"],
    ) -> YRTransportMetadata:
    ...
```

## Running Tests

All test programs are located in the `tests/` directory and are built on the pytest
framework.

```bash
# run tests
pip install -e ".[all]"
pytest -v
```

## Sign the Ascend CLA

When submitting a PR for the first time, please sign the Ascend
[CLA (Contributor License Agreement)](https://clasign.osinfra.cn/sign/690ca9ddf91c03dee6082ab1).
The email address used to sign the CLA must match your Git commit signature.
