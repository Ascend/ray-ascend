# Performance Benchmark Report

> Last updated: 2026/05/08

This report documents the performance benchmark results for Ray-Ascend's core data
transport capabilities, focusing on post-training RL sample transmission and training-
inference weight synchronization.

## Test Environment

### Hardware

| Component | Specification |
| --------- | ------------- |
| NPU Model | _e.g., Ascend 910B_ |
| Node Count | _e.g., 2 nodes_ |
| Network | _e.g., 100 Gbps RoCE_ |
| Memory | _e.g., 64 GB_ |

### Software

| Component | Version |
| --------- | ------- |
| CANN | _e.g., 8.0.RC1_ |
| Ray | _e.g., 2.9.0_ |
| Python | _e.g., 3.10_ |
| Ray-Ascend | _e.g., commit hash or version_ |


______________________________________________________________________

## 1. RL Samples Transmission

This benchmark compares the performance of YR Direct Transport (RDT) against Ray's
default serialization for NPU tensor transmission in post-training RL scenarios.
### 1.1 Local mode

#### 1.1.1 Base Configuration
```yaml
backend: yr
init_mode: metastore
placement: local
device: npu
warmup_times: 5
count: 20
```
### 1.2 Results

#### Throughput Comparison

| Tensor Size | YR RDT (Gb/s) | Ray Serialization (Gb/s) | Speedup |
| ----------- | ------------- | ------------------------- | ------- |
| 1 KB        |               |                           |         |
| 64 KB       |               |                           |         |
| 1 MB        |               |                           |         |
| 16 MB       |               |                           |         |
| 64 MB       |               |                           |         |

#### Latency Comparison (P50 / P99)

| Tensor Size | YR RDT (ms) | Ray Serialization (ms) | Reduction |
| ----------- | ----------- | ----------------------- | --------- |
| 1 KB        |             |                         |           |
| 64 KB       |             |                         |           |
| 1 MB        |             |                         |           |
| 16 MB       |             |                         |           |
| 64 MB       |             |                         |           |

### 1.3 Analysis

_TODO: Add analysis of results_

- Performance characteristics at different tensor sizes
- Bottlenecks identified
- Recommendations for optimal tensor sizes

______________________________________________________________________

## 2. Weight Synchronization (P2P Transfer)

> **Status**: Work in Progress
>
> This section will be updated with benchmark results after testing is complete.

### 2.1 Test Scenario

**Use Case**: Synchronizing model weights between training and inference instances
in a training-inference co-located deployment.

**Data Characteristics**:

- Model parameter sizes: _e.g., 7B, 13B, 70B parameters_
- Data type: _e.g., float16 / bfloat16_
- Typical size range: _e.g., 14GB - 140GB_

### 2.2 Planned Tests

- [ ] Single-node P2P weight transfer
- [ ] Cross-node P2P weight transfer
- [ ] Incremental weight sync (delta updates)
- [ ] Concurrent weight sync with training

### 2.3 Results

_TBD after testing_

______________________________________________________________________

## 3. Conclusions

### Key Findings

1. _Summary of RL samples transmission results_
2. _Summary of weight synchronization results (after testing)_

### Recommendations

1. _When to use YR RDT vs Ray serialization_
2. _Optimal configurations for different scenarios_

______________________________________________________________________

## Appendix

### Test Reproduction

```bash
# RL Samples Transmission Test
python tests/benchmarks/direct_transport_perftest.py \
  --backend yr \
  --device npu \
  --tensor-size-kb 1024 \
  --warmup-times 5 \
  --count 100
```

### Raw Data

_Link to detailed raw data files or embedded data tables if needed_