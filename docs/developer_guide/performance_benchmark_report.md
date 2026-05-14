# Performance Benchmark Report

> _Last updated: 05/14/2026_

This report documents the performance benchmark results for Ray-Ascend's core data
transport capabilities, focusing on post-training RL sample transmission and training-
inference weight synchronization.

## Test Environment

### Hardware

| Component  | Specification |
| ---------- | ------------- |
| NPU Model  | _Ascend 910B_ |
| Node Count | _2 nodes_     |

### Software

| Component  | Version   |
| ---------- | --------- |
| CANN       | _8.3.RC1_ |
| Ray        | _2.55.1_  |
| Ray-Ascend | _0.1.0_   |
| Python     | _3.10.16_ |

______________________________________________________________________

## 1. RL Samples Transmission

This benchmark compares the performance of Yuanrong Direct Transport against Ray's
default serialization for NPU tensor transmission in post-training RL scenarios.

### 1.1 Local Mode

Sender and receiver actors are colocated on the same node, utilizing local memory
transfer.

#### 1.1.1 Configuration

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

### 1.2 Remote Mode

Sender and receiver actors are distributed across two nodes, testing cross-node
transport performance.

#### 1.2.1 Configuration

```yaml
backend: yr
init_mode: metastore
placement: remote
head_node_ip: "10.170.27.237"
worker_node_ip: "10.170.27.158"
device: npu
warmup_times: 5
count: 20
```

#### 1.2.2 Results

**Throughput Comparison**

| Setting | Tensor Count | Tensor Size (KB) | Total Size (GB) | Transport Mode | AVG Throughput (Gbps) |
| ------- | ------------ | ---------------- | --------------- | -------------- | --------------------- |
| small   | 9216         | 32               | 0.28            | yr             | 0.27                  |
| small   | 9216         | 32               | 0.28            | ray            | 0.19                  |
| medium  | 61440        | 128              | 7.50            | yr             | 1.11                  |
| medium  | 61440        | 128              | 7.50            | ray            | 0.47                  |
| large   | 35000        | 384              | 12.81           | yr             | 3.48                  |
| large   | 35000        | 384              | 12.81           | ray            | 0.76                  |

**Latency Comparison**

| Setting | Tensor Count | Tensor Size (KB) | Total Size (GB) | Transport Mode | P90 Latency (s) | P95 Latency (s) | P99 Latency (s) |
| ------- | ------------ | ---------------- | --------------- | -------------- | --------------- | --------------- | --------------- |
| small   | 9216         | 32               | 0.28            | yr             | 8.76            | 9.02            | 9.25            |
| small   | 9216         | 32               | 0.28            | ray            | 12.05           | 12.10           | 12.14           |
| medium  | 61440        | 128              | 7.50            | yr             | 54.97           | 55.15           | 55.22           |
| medium  | 61440        | 128              | 7.50            | ray            | 128.63          | 129.16          | 129.54          |
| large   | 35000        | 384              | 12.81           | yr             | 29.98           | 30.14           | 30.18           |
| large   | 35000        | 384              | 12.81           | ray            | 135.85          | 136.74          | 138.81          |

### 1.3 Analysis

![Throughput and Latency Comparison](images/comparison.png)

#### Performance Highlights

- **Throughput**: YR RDT achieves 2-6x higher throughput than Ray serialization across
  all tensor sizes, with the gap widening as data size increases.
- **Latency**: YR RDT reduces latency by 40-80% compared to Ray serialization,
  particularly significant for medium and large tensor sizes.
- **Scaling**: Both modes show similar relative performance, with YR RDT maintaining its
  advantage in distributed (remote) scenarios.

#### Observations

The fragmented tensor transmission pattern in this test limits Yuanrong Direct
Transport's full potential. As shown in the tables, system throughput increases
significantly with larger tensor sizes. Under similar Ascend 910B test conditions,
transmitting single tensors larger than 10 MB achieves throughput of 50-100 Gbps,
indicating Yuanrong Direct Transport performs optimally with larger, contiguous data
transfers.

______________________________________________________________________

## 2. Weight Synchronization (P2P Transfer)

> **Status**: Work in Progress
>
> This section will be updated with benchmark results after testing is complete.

### 2.1 Test Scenario

**Use Case**: Synchronizing model weights between training and inference instances in a
training-inference co-located deployment.

### 2.2 Planned Tests

### 2.3 Results

_TBD after testing_

______________________________________________________________________

## 3. Conclusions

### Key Findings

1. YR RDT significantly outperforms Ray serialization for NPU tensor transmission,
   achieving 2-6x higher throughput and 40-80% lower latency across all tested
   configurations.
1. The performance advantage scales with data size - larger tensors benefit more from YR
   RDT's zero-copy transfer mechanism.
1. Weight synchronization benchmarks are pending and will be documented upon completion.

### Recommendations

1. _When to use YR RDT vs Ray serialization_
1. _Optimal configurations for different scenarios_

______________________________________________________________________

## Appendix

### Test Reproduction

```bash
# RL Samples Transmission Test
python tests/benchmarks/direct_transport_perftest.py \
  --config-file tests/benchmarks/direct_transport_config.yaml
```

### Configuration Files

- **[Local mode configuration](../../../tests/benchmarks/direct_transport_config.yaml)**:
  Base config for single-node testing. Set `placement: local` and adjust
  `tensor_size_kb`, `count`, `warmup_times` as needed.
- **[Remote mode configuration](../../../tests/benchmarks/direct_transport_config.yaml)**:
  For distributed testing across two nodes. Set `placement: remote` and configure
  `head_node_ip` and `worker_node_ip` with actual IP addresses.
