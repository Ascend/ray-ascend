# Contributing

## Instructions for Contribution
### Code style

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

### Install pre-commit
```
pip install pre-commit
# launch pre-commit (pre-commit is triggered when you use `git commit`.)
pre-commit install
```

## A new ccl

## A new transport