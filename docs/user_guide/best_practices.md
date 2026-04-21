# Best Practices, Troubleshooting & FAQ

> _Last updated: 03/24/2026_

## Best Practices

### HCCL Collective Communication

1. **Device Consistency**: Ensure all tensors used in collective operations reside on
   the same NPU device that was used during communicator initialization.

1. **Group Cleanup**: Always call `destroy_group()` when done to free communicator
   resources.

1. **Rank Coordination**: All ranks must participate in collective operations in the
   same order.

### YR Direct Transport

1. **Environment Setup**: Always set `YR_DS_WORKER_HOST` and `YR_DS_WORKER_PORT` before
   using YR transport.

1. **Single Device Type**: All tensors in one RDT object must have the same device type
   (all CPU or all NPU).

1. **Decorator Usage**: Use `@ray.method(tensor_transport="YR")` on actor methods that
   return or receive tensors to be transported.

1. **Health Check**: Use `actor_has_tensor_transport()` to verify an actor has YR
   transport properly configured before sending tensors.

1. **Zero-Copy for CPU**: CPU tensors use a packed binary format for efficient zero-copy
   transfer via shared memory.

## Troubleshooting

### HCCL Issues

**Problem**: "Unable to meet other processes at the rendezvous store"

**Solution**:

- Ensure all ranks are calling `create_collective_group()` with the same group name
- Verify all actors have the correct resources assigned (`resources={"NPU": 1}`)
- Check that the same number of actors are specified in `create_collective_group()`

**Problem**: "Collective ops must use the same device as communicator initialization"

**Solution**: Ensure the tensor you're passing is on the same NPU device that was
current when the `HCCLGroup` was created.

### YR Transport Issues

**Problem**: "YR DS worker env not set"

**Solution**: Set the required environment variables:

```bash
export YR_DS_WORKER_HOST="127.0.0.1"
export YR_DS_WORKER_PORT="31502"
```

**Problem**: "Missing optional dependency 'datasystem'"

**Solution**: Install the YR extras:

```bash
pip install "ray-ascend[yr]"
```

**Problem**: "Failed to initialize YR DS client"

**Solution**:

- Verify the YR DataSystem worker is running at the specified host and port
- Check network connectivity
- Verify etcd is running and accessible

## FAQ

**Q: Can I use ray-ascend without NPUs?**

A: Yes! The YR transport supports CPU tensors without requiring NPU hardware or CANN.
Just install `ray-ascend[yr]` and use CPU tensors.

**Q: What Ray versions are compatible?**

A: ray-ascend should be used with the same Ray version. Check the release notes for
specific version compatibility.

**Q: Do I need to use both HCCL and YR transport together?**

A: No, they can be used independently. Use HCCL for collective operations, YR for direct
point-to-point tensor transfer, or both together as needed.

**Q: How do I choose between HCCL send/recv and YR transport?**

A: Use HCCL send/recv when you already have a collective group set up and need
fine-grained point-to-point within that group. Use YR transport for general-purpose
tensor transfer between any actors, with built-in zero-copy optimization.

## Contributing

For information on contributing to ray-ascend, see the
[Developer Guide](../developer_guide/index.md) and
[Contributing Guide](../developer_guide/contributing.md).
