# API Reference

> _Last updated: 03/24/2026_

## HCCLGroup

The main class for HCCL collective communication.

### Constructor

```python
HCCLGroup(world_size: int, rank: int, group_name: str)
```

### Methods

| Method                                                      | Description                     |
| ----------------------------------------------------------- | ------------------------------- |
| `broadcast(tensor, broadcast_options)`                      | Broadcast tensor from root rank |
| `allreduce(tensor, allreduce_options)`                      | All-reduce tensor across group  |
| `allgather(tensor_list, tensor, allgather_options)`         | Gather tensors from all ranks   |
| `reduce(tensor, reduce_options)`                            | Reduce tensor to root rank      |
| `reducescatter(tensor, tensor_list, reducescatter_options)` | Reduce and scatter              |
| `send(tensor, send_options)`                                | Send tensor to peer             |
| `recv(tensor, recv_options)`                                | Receive tensor from peer        |
| `barrier(barrier_options)`                                  | Synchronize all ranks           |
| `destroy_group()`                                           | Clean up communicator resources |

## YRTensorTransport

The main class for YR direct tensor transport.

### Constructor

```python
YRTensorTransport()
```

### Environment Variables

| Variable            | Description                                         |
| ------------------- | --------------------------------------------------- |
| `YR_DS_WORKER_HOST` | Host address of the YR DataSystem worker (required) |
| `YR_DS_WORKER_PORT` | Port of the YR DataSystem worker (required)         |

### Methods

| Method                                                                            | Description                               |
| --------------------------------------------------------------------------------- | ----------------------------------------- |
| `tensor_transport_backend()`                                                      | Returns "YR"                              |
| `is_one_sided()`                                                                  | Returns True (one-sided communication)    |
| `get_ds_client(device_type)`                                                      | Get or create the DataSystem client       |
| `actor_has_tensor_transport(actor)`                                               | Check if actor has YR transport available |
| `extract_tensor_transport_metadata(obj_id, gpu_object)`                           | Extract metadata for transport            |
| `recv_multiple_tensors(obj_id, tensor_transport_metadata, communicator_metadata)` | Receive tensors                           |
| `garbage_collect(obj_id, tensor_transport_meta)`                                  | Clean up resources                        |

### Registration

Register YR tensor transport with:

```python
from ray_ascend import register_yr_tensor_transport

register_yr_tensor_transport(["npu", "cpu"])
```
