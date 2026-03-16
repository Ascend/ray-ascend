import logging
from abc import ABC, abstractmethod
from typing import Optional


class RayAscendBaseTester(ABC):
    @abstractmethod
    def run_test(self):
        raise NotImplementedError("run_test must be implemented by subclasses.")

    @abstractmethod
    def close(self):
        raise NotImplementedError("close must be implemented by subclasses.")

    @staticmethod
    def calculate_latency_percentiles(latencies: list[float]) -> dict[str, float]:
        """
        Calculate latency percentiles (P50, P75, P90, P95, P99).

        Args:
            latencies: List of latency measurements in seconds.

        Returns:
            Dictionary containing percentile values with keys:
            'p50', 'p75', 'p90', 'p95', 'p99'
        """
        if not latencies:
            raise ValueError("Latencies list cannot be empty")

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            """Calculate the p-th percentile.

            Uses linear interpolation for small sample sizes (n < 30)
            and nearest-rank method for large sample sizes (n >= 30).
            """
            if n < 30:
                # Linear interpolation: smooths data for small sample sizes
                k = (n - 1) * p / 100.0
                f = int(k)
                c = f + 1 if f + 1 < n else f
                d = k - f
                return sorted_latencies[f] + d * (
                    sorted_latencies[c] - sorted_latencies[f]
                )
            else:
                # Nearest-rank method: returns actual measured sample for large sample sizes
                k = int((n - 1) * p / 100.0)
                return sorted_latencies[k]

        return {
            "p50": percentile(50),
            "p75": percentile(75),
            "p90": percentile(90),
            "p95": percentile(95),
            "p99": percentile(99),
        }

    @staticmethod
    def calculate_throughput(
        total_data_size_gb: float,
        times: list[float],
        return_stats: bool = True,
    ) -> dict[str, float] | float:
        """
        Calculate throughput statistics.

        Args:
            total_data_size_gb: Total data size transferred in GB.
            times: List of time measurements in seconds for each transfer.
            return_stats: If True, return detailed statistics dict.
                         If False, return only average throughput.

        Returns:
            If return_stats is True:
                Dictionary with keys:
                - 'avg_throughput_gbps': Average throughput in Gb/s
                - 'min_throughput_gbps': Minimum throughput in Gb/s
                - 'max_throughput_gbps': Maximum throughput in Gb/s
                - 'total_time_s': Total time across all iterations in seconds
                - 'total_iterations': Number of iterations
            If return_stats is False:
                Average throughput in Gb/s (float)
        """
        if not times:
            raise ValueError("Times list cannot be empty")
        if total_data_size_gb <= 0:
            raise ValueError("Total data size must be positive")

        # Calculate throughput for each iteration (in Gb/s)
        throughputs = [(total_data_size_gb * 8) / t for t in times]

        if not return_stats:
            return sum(throughputs) / len(throughputs)

        total_time = sum(times)

        return {
            "avg_throughput_gbps": sum(throughputs) / len(throughputs),
            "min_throughput_gbps": min(throughputs),
            "max_throughput_gbps": max(throughputs),
            "total_time_s": total_time,
            "total_iterations": len(times),
        }

    @staticmethod
    def log_performance_summary(
        *,
        logger: logging.Logger,
        test_name: str,
        total_data_size_gb: float,
        iterations: int,
        latency_percentiles: Optional[dict[str, float]] = None,
        throughput_stats: Optional[dict[str, float]] = None,
    ) -> None:
        """
        Log comprehensive performance summary.

        Args:
            logger: Logger instance for output.
            test_name: Name of the test (e.g., "YR LOCAL").
            total_data_size_gb: Total data size in GB.
            iterations: Number of iterations executed.
            latency_percentiles: Dict with keys 'p50', 'p75', 'p90', 'p95', 'p99'.
            throughput_stats: Dict from calculate_throughput() with
                            'avg_throughput_gbps', 'min_throughput_gbps',
                            'max_throughput_gbps', 'total_time_s', 'total_iterations'.
        """
        logger.info("=" * 60)
        logger.info(f"{test_name} BANDWIDTH TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Data Size: {total_data_size_gb:.6f} GB")
        logger.info(f"Number of Iterations: {iterations}")

        if throughput_stats:
            # Log throughput stats
            logger.info(
                f"Average Transport Time: {throughput_stats['total_time_s'] / throughput_stats['total_iterations']:.8f}s"
            )
            logger.info(
                f"Average Transport Throughput: {throughput_stats['avg_throughput_gbps']:.8f} Gb/s"
            )
            logger.info(
                f"Min Transport Throughput: {throughput_stats['min_throughput_gbps']:.8f} Gb/s"
            )
            logger.info(
                f"Max Transport Throughput: {throughput_stats['max_throughput_gbps']:.8f} Gb/s"
            )

        if latency_percentiles:
            # Log latency percentiles
            logger.info(f"P50 Latency: {latency_percentiles['p50']:.8f}s")
            logger.info(f"P75 Latency: {latency_percentiles['p75']:.8f}s")
            logger.info(f"P90 Latency: {latency_percentiles['p90']:.8f}s")
            logger.info(f"P95 Latency: {latency_percentiles['p95']:.8f}s")
            logger.info(f"P99 Latency: {latency_percentiles['p99']:.8f}s")
