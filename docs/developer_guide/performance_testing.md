# Performance Testing

> Last updated: 03/30/2026

This guide explains how to run performance tests for ray-ascend, including YuanRong (YR)
Direct Transport and HCCL collective communication performance tests.

## Overview

The performance test suite is located in `tests/benchmarks/` and supports the following
test types:

- **YR Direct Transport**: Tests YuanRong direct tensor transport performance
- **HCCL Collective Communication**: Tests HCCL collective operations performance
  (Coming Soon)

All tests support the following common configurations:

- **Placement Modes**: Local (single node) and Remote (distributed across nodes)
- **Devices**: NPU and CPU tensors

## Common Prerequisites

### Required Dependencies

```bash
# Install ray-ascend with all features
pip install -e ".[all]"
```

### For Remote Mode Testing

Set up a Ray cluster with two nodes:

**Head Node**:

```bash
ray start --head --resources='{"node:<HEAD_IP>": 1}'
```

**Worker Node**:

```bash
ray start --address <HEAD_IP>:6379 --resources='{"node:<WORKER_IP>": 1}'
```

Replace `<HEAD_IP>` and `<WORKER_IP>` with actual IP addresses.

______________________________________________________________________

## YR Direct Transport Performance Test

### Overview

Tests YuanRong direct tensor transport performance between Ray actors. This test
measures the throughput and latency of tensor transfers using YR Direct Transport.

### Running the Test

#### Using Command Line Arguments

```bash
python tests/benchmarks/direct_transport_perftest.py \
  --backend yr \
  --placement local \
  --device cpu \
  --tensor-size-kb 1024 \
  --warmup-times 2 \
  --count 5
```

#### Using Configuration File

Create a YAML configuration file (e.g., `config.yaml`):

```yaml
# Transport backend: 'yr' for YR Direct Transport, 'hccl' for HCCL
backend: yr
# Test deployment mode: 'local' (same node) or 'remote' (distributed)
placement: remote
# Device to run tensors on: 'npu' or 'cpu'
device: npu
# IP address of the Ray head node (required for remote mode)
head_node_ip: "10.170.27.237"
# IP address of the worker node (required for remote mode)
worker_node_ip: "10.170.27.158"
# Total tensor size in KB
tensor_size_kb: 1000
# Number of warmup iterations before measurement
warmup_times: 5
# Number of iterations for the actual test (results are averaged)
count: 100
```

Then run:

```bash
python tests/benchmarks/direct_transport_perftest.py --config-file config.yaml
```

Command-line arguments override config file settings.

### Parameters

| Parameter          | Type | Choices       | Default  | Description                                                  |
| ------------------ | ---- | ------------- | -------- | ------------------------------------------------------------ |
| `--backend`        | str  | yr, hccl      | required | Transport backend                                            |
| `--placement`      | str  | local, remote | local    | Test deployment mode                                         |
| `--device`         | str  | npu, cpu      | cpu      | Device to run tensors on                                     |
| `--head-node-ip`   | str  | -             | -        | IP address of Ray head node (required for remote mode)       |
| `--worker-node-ip` | str  | -             | -        | IP address of worker node (required for remote mode)         |
| `--tensor-size-kb` | int  | -             | 1024     | Total number of KB in tensor (converted to float32 elements) |
| `--warmup-times`   | int  | -             | 2        | Number of warmup iterations before measurement               |
| `--count`          | int  | -             | 5        | Number of test iterations (results are averaged)             |
| `--config-file`    | str  | -             | -        | Path to YAML config file                                     |

### Example Configuration

An example configuration file is provided at
`tests/benchmarks/direct_transport_config.yaml`.

______________________________________________________________________

## HCCL Collective Communication Performance Test

### Overview

Tests HCCL collective operations performance (all-reduce, all-gather, broadcast, etc.)
across multiple workers.

> **Note**: This test is under development and will be available in future releases.

______________________________________________________________________

## Common Test Output

All performance tests output the following metrics:

- **Latency Percentiles**: P50, P75, P90, P95, P99
- **Throughput Statistics**: Average, Min, Max throughput in Gb/s
- **Total Data Size**: In GB
- **Number of Iterations**: Test iterations performed

### Example Output

```
============================================================
YR LOCAL BANDWIDTH TEST SUMMARY
============================================================
Total Data Size: 0.001024 GB
Number of Iterations: 5
Average Transport Time: 0.00012345s
Average Transport Throughput: 66.35714286 Gb/s
Min Transport Throughput: 60.00000000 Gb/s
Max Transport Throughput: 70.00000000 Gb/s
P50 Latency: 0.00012000s
P75 Latency: 0.00012500s
P90 Latency: 0.00012800s
P95 Latency: 0.00013000s
P99 Latency: 0.00013200s
```

______________________________________________________________________

## Architecture

The performance test framework consists of:

- `base_perftest.py`: Abstract base class with common test infrastructure
- `direct_transport_perftest.py`: YR Direct Transport specific implementation
- `direct_transport_config.yaml`: Example configuration file

### Test Flow

1. Initialize etcd service for coordination
1. Start DataSystem actors on target nodes
1. Create test actors (sender and receiver)
1. Warm-up iterations
1. Run performance measurements
1. Calculate and display statistics
1. Cleanup resources
