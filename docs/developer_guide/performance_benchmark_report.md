# Performance Benchmark Report

> Last updated: 2026/05/08

This report documents the performance benchmark results for Ray-Ascend's core data
transport capabilities, focusing on post-training RL sample transmission and training-
inference weight synchronization.

## Test Environment

### Hardware

| Component  | Specification |
| ---------- | ------------- |
| NPU Model  | _Ascend 910B_ |
| Node Count | _2 nodes_     |
| Network    | \_\_          |

### Software

| Component  | Version   |
| ---------- | --------- |
| CANN       | _8.3.RC1_ |
| Ray        | _2.55.1_  |
| Python     | _3.10.16_ |
| Ray-Ascend | _0.1.0_   |

______________________________________________________________________

## 1. RL Samples Transmission

This benchmark compares the performance of YR Direct Transport (RDT) against Ray's
default serialization for NPU tensor transmission in post-training RL scenarios.

### 1.1 Local mode

Colocate Ray actors on the same node:

#### 1.1.1 Base Configuration

```yaml
backend: yr
init_mode: metastore
placement: local
device: npu
warmup_times: 5
count: 20
```

#### 1.1.2 Results

**Throughput Comparison**

| Setting | Tensor Count | Tensor Size (KB) | Total Size (GB) | Transport Mode | AVG Throughput (Gbps) |
| :------ | :----------- | :--------------- | :-------------- | :------------- | :-------------------- |
| small   | 9216         | 32               | 0.28            | yr             | 0.35                  |
| small   | 9216         | 32               | 0.28            | ray            | 0.19                  |
| medium  | 61440        | 128              | 7.50            | yr             | 1.25                  |
| medium  | 61440        | 128              | 7.50            | ray            | 0.42                  |
| large   | 35000        | 384              | 12.81           | yr             | 4.27                  |
| large   | 35000        | 384              | 12.81           | ray            | 0.70                  |

**Latency Comparison**

| Setting | Tensor Count | Tensor Size (KB) | Total Size (GB) | Transport Mode | P90 Latency (s) | P95 Latency (s) | P99 Latency (s) |
| :------ | :----------- | :--------------- | :-------------- | :------------- | :-------------- | :-------------- | :-------------- |
| small   | 9216         | 32               | 0.28            | yr             | 7.10            | 7.30            | 7.33            |
| small   | 9216         | 32               | 0.28            | ray            | 12.14           | 12.15           | 12.27           |
| medium  | 61440        | 128              | 7.50            | yr             | 48.99           | 49.09           | 49.17           |
| medium  | 61440        | 128              | 7.50            | ray            | 144.05          | 144.54          | 152.03          |
| large   | 35000        | 384              | 12.81           | yr             | 24.38           | 24.46           | 24.76           |
| large   | 35000        | 384              | 12.81           | ray            | 148.14          | 148.90          | 149.16          |

### 1.2 Remote mode

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

**Use Case**: Synchronizing model weights between training and inference instances in a
training-inference co-located deployment.

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
1. _Summary of weight synchronization results (after testing)_

### Recommendations

1. _When to use YR RDT vs Ray serialization_
1. _Optimal configurations for different scenarios_

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
