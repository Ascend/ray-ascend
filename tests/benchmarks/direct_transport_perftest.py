import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Union

import ray
import torch
import yaml
from torch import Tensor

from ray_ascend import register_yr_tensor_transport
from ray_ascend.utils import cleanup_yr_resources, start_etcd

# Add parent directory to sys.path for importing base_perftest
sys.path.insert(0, str(Path(__file__).parent))

from base_perftest import RayAscendBaseTester

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


HEAD_NODE_IP = "NodeA"
WORKER_NODE_IP = "NodeB"


class EtcdUtil:
    """Utility class to manage etcd process and lifecycle.

    Used in CI testing scenarios where etcd needs to be started automatically.
    """

    def __init__(self, host: str = "127.0.0.1"):
        """Start etcd process.

        Args:
            host: The host to bind etcd to. Defaults to "127.0.0.1".
        """
        self.etcd_addr, self.etcd_proc, self.etcd_data_dir = start_etcd(host=host)
        logger.info(f"EtcdUtil initialized with address: {self.etcd_addr}")

    def close(self):
        """Stop etcd process and clean up resources."""
        if self.etcd_proc:
            try:
                self.etcd_proc.terminate()
                self.etcd_proc.wait(timeout=5)
                logger.info("Etcd process terminated successfully")
                self.etcd_proc = None
            except Exception as e:
                logger.error(f"Error terminating etcd process: {e}")

        if self.etcd_data_dir and os.path.exists(self.etcd_data_dir):
            try:
                shutil.rmtree(self.etcd_data_dir, ignore_errors=True)
                logger.info(f"Etcd data directory cleaned up: {self.etcd_data_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up etcd data directory: {e}")

    def __del__(self):
        """Ensure etcd is closed when object is destroyed."""
        try:
            self.close()
        except Exception:
            pass


def check_npu_is_available():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        raise ImportError(
            "torch_npu is not installed. Please install it to use NPU device."
        )
    else:
        if not torch.npu.is_available():
            raise RuntimeError(
                "NPU device specified but not available. Please check your environment."
            )


def load_config_from_file(config_file: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config or {}


def parse_args() -> argparse.Namespace:
    """
    The following parameters are not currently supported:
    transport
    output-format
    """
    arg_configs = [
        {
            "name": "--backend",
            "type": str,
            "choices": ["yr", "hccl"],
            "help": "Transport backend: 'yr' for YR Direct Transport, 'hccl' for HCCL.",
        },
        {
            "name": "--init-mode",
            "type": str,
            "choices": ["etcd", "metastore"],
            "default": "etcd",
            "help": (
                "YR backend initialization mode. \n"
                "'etcd': manual etcd setup (default). \n"
                "'metastore': auto-init via YRBackendCoordinator."
            ),
        },
        {
            "name": "--placement",
            "type": str,
            "choices": ["local", "remote"],
            "default": "local",
            "help": (
                "Test deployment mode. \n"
                "'local': all actors run on the same node (default). \n"
                "'remote': actors are distributed across multiple nodes. \n"
                "To use 'remote', first set up a Ray cluster: \n"
                "on head node: `ray start --head --resources='{\"node:<HEAD_IP>\": 1}'`; \n"
                "on worker node: `ray start --address <HEAD_IP>:6379 --resources='{\"node:<WORKER_IP>\": 1}'`. \n"
                "Replace <HEAD_IP> and <WORKER_IP> with actual IPs."
            ),
        },
        {
            "name": "--device",
            "type": str,
            "choices": ["npu", "cpu"],
            "default": "cpu",
            "help": "Device to run tensors on: 'npu' or 'cpu'.",
        },
        {
            "name": "--head-node-ip",
            "type": str,
            "help": "IP address of the Ray head node. Required in 'remote' mode; driver must run on head node.",
        },
        {
            "name": "--worker-node-ip",
            "type": str,
            "help": "IP address of the worker node. Required in 'remote' mode.",
        },
        {
            "name": "--tensor-count",
            "type": int,
            "default": 1,
            "help": "Number of tensors to transport in the list.",
        },
        {
            "name": "--tensor-size-kb",
            "type": int,
            "default": 1024,
            "help": "Size of each tensor in KB.",
        },
        {
            "name": "--warmup-times",
            "type": int,
            "default": 2,
            "help": "Number of warmup iterations before measurement (default: 3).",
        },
        {
            "name": "--config-file",
            "type": str,
            "help": (
                "Path to a YAML config file with test parameters. "
                "Command-line arguments override config file settings."
            ),
        },
        {
            "name": "--count",
            "type": int,
            "default": 5,
            "help": "Number of iterations for the actual test (default: 1). Results are averaged.",
        },
    ]

    parser = argparse.ArgumentParser()
    for arg in arg_configs:
        arg_copy = {k: v for k, v in arg.items() if k != "name"}
        parser.add_argument(arg["name"], **arg_copy)  # type: ignore[arg-type]

    args_partial, _ = parser.parse_known_args()

    # load config file to defaults if provided, so that command-line args can override them
    config_defaults = {}
    if args_partial.config_file:
        try:
            config_defaults = load_config_from_file(args_partial.config_file)
        except (FileNotFoundError, yaml.YAMLError) as e:
            parser.error(str(e))

    # set defaults and parse final args
    parser.set_defaults(**config_defaults)
    final_args = parser.parse_args()
    if final_args.backend is None:
        parser.error("--backend is required")
    if final_args.placement == "remote":
        if not final_args.head_node_ip:
            parser.error("--head-node-ip is required when --placement=remote")
        if not final_args.worker_node_ip:
            parser.error("--worker-node-ip is required when --placement=remote")
        global HEAD_NODE_IP, WORKER_NODE_IP
        HEAD_NODE_IP = final_args.head_node_ip
        WORKER_NODE_IP = final_args.worker_node_ip
    if final_args.count <= 0:
        parser.error("--count must be a positive integer")
    if final_args.warmup_times < 0:
        parser.error("--warmup-times cannot be negative")
    if final_args.tensor_size_kb <= 0:
        parser.error("--tensor-size-kb must be a positive integer")
    if final_args.tensor_count <= 0:
        parser.error("--tensor-count must be a positive integer")

    logger.info(
        f"Test configuration:\n{yaml.dump(vars(final_args), default_flow_style=False)}"
    )
    return final_args


def _create_yr_tensor_transport_actor_class():
    """Factory function to create YRTensorTransportActor class dynamically.

    This must be called after register_yr_tensor_transport() to ensure
    YR transport is registered before the @ray.method(tensor_transport="YR")
    decorator is parsed.
    """

    @ray.remote
    class YRTensorTransportActor:

        def __init__(self, config: argparse.Namespace, node_ip: Optional[str] = None):
            # Register transport in actor process
            register_yr_tensor_transport(["npu", "cpu"])
            self.config = config
            self.node_ip = node_ip
            self.data: Optional[Union[Tensor, list[Tensor]]] = None

            if self.config.device == "npu":
                check_npu_is_available()

        def generate_tensor(self):
            # convert KB to number of float32 elements
            seq_len = self.config.tensor_size_kb * 1000 // 4
            self.data = (
                torch.randn(
                    seq_len,
                    device=self.config.device,
                )
                if self.config.tensor_count == 1
                else [
                    torch.randn(
                        seq_len,
                        device=self.config.device,
                    )
                    for _ in range(self.config.tensor_count)
                ]
            )
            if isinstance(self.data, Tensor):
                logger.info(f"Generated tensor of shape {self.data.shape}")
            else:
                logger.info(
                    f"Generated {len(self.data)} tensors, each of shape {self.data[0].shape}"
                )

        @ray.method(tensor_transport="YR")
        def transport_tensor_via_yr(self) -> Union[Tensor, list[Tensor]]:
            if self.data is None:
                raise RuntimeError(
                    "Tensor not generated yet. Call generate_tensor first."
                )
            return self.data

        def recv_tensor(self, data: Union[Tensor, list[Tensor]]) -> bool:
            if isinstance(data, Tensor) or (
                isinstance(data, list) and all(isinstance(t, Tensor) for t in data)
            ):
                return True
            return False

    return YRTensorTransportActor


class HCCLTransportTester:
    pass


class YRDirectTransportTester(RayAscendBaseTester):
    """Tester for YR direct transport using register_yr_tensor_transport API.

    Args:
        config: Test configuration from argparse
        init_mode: Initialization mode, "etcd" or "metastore"
        remote_mode: Deployment mode, "local" or "remote"
    """

    def __init__(
        self,
        config: argparse.Namespace,
        init_mode: str = "etcd",
        remote_mode: str = "local",
    ):
        self.config = config
        self.init_mode = init_mode
        self.remote_mode = remote_mode

        # etcd utility (only used when auto-starting etcd in CI scenarios)
        self.etcd_util: Optional[EtcdUtil] = None

        # Actor class (created after YR transport registration)
        self._actor_class: Optional[type] = None

        if self.config.device == "npu":
            check_npu_is_available()

        self._setup_environment()

        # Create actor class after YR transport is registered
        self._actor_class = _create_yr_tensor_transport_actor_class()

    def _setup_environment(self):
        """Set environment variables and initialize YR backend."""
        # Set init mode
        os.environ["YR_DS_INIT_MODE"] = self.init_mode

        if self.init_mode == "etcd":
            # Etcd mode: start etcd if not provided
            etcd_address = os.getenv("YR_DS_ETCD_ADDRESS")
            if etcd_address:
                logger.info(f"Using user-provided etcd at {etcd_address}")
            else:
                # CI scenario: auto-start etcd
                etcd_host = (
                    HEAD_NODE_IP if self.remote_mode == "remote" else "127.0.0.1"
                )
                self.etcd_util = EtcdUtil(host=etcd_host)
                os.environ["YR_DS_ETCD_ADDRESS"] = self.etcd_util.etcd_addr
                logger.info(f"Auto-started etcd for CI at {self.etcd_util.etcd_addr}")

        elif self.init_mode == "metastore":
            # Metastore mode: set ports if not already set
            if "YR_DS_WORKER_PORT" not in os.environ:
                os.environ["YR_DS_WORKER_PORT"] = "31501"
            if "YR_DS_METASTORE_PORT" not in os.environ:
                os.environ["YR_DS_METASTORE_PORT"] = "2379"

        # Set worker args if not already set
        if "YR_DS_WORKER_ARGS" not in os.environ:
            os.environ["YR_DS_WORKER_ARGS"] = "--shared_memory_size_mb 2048"

        logger.info(
            f"Initializing YR backend in {self.init_mode} mode "
            f"(remote_mode={self.remote_mode})..."
        )

        # Initialize YR backend via register_yr_tensor_transport
        register_yr_tensor_transport(["npu", "cpu"])
        logger.info(f"YR {self.init_mode} backend initialized")

    def _initialize_test_actor(
        self,
    ) -> tuple[ray.actor.ActorHandle, ray.actor.ActorHandle]:
        if self._actor_class is None:
            raise RuntimeError("Actor class not initialized")

        if self.remote_mode == "remote":
            sender_actor = self._actor_class.options(  # type: ignore[attr-defined]
                resources={f"node:{HEAD_NODE_IP}": 0.001, "NPU": 1}
            ).remote(self.config, HEAD_NODE_IP)

            receiver_actor = self._actor_class.options(  # type: ignore[attr-defined]
                resources={f"node:{WORKER_NODE_IP}": 0.001, "NPU": 1}
            ).remote(self.config, WORKER_NODE_IP)

        else:
            sender_actor = self._actor_class.options(resources={"NPU": 1}).remote(  # type: ignore[attr-defined]
                self.config
            )
            receiver_actor = self._actor_class.options(resources={"NPU": 1}).remote(  # type: ignore[attr-defined]
                self.config
            )

        return sender_actor, receiver_actor

    def run_test(self):
        sender_actor, receiver_actor = self._initialize_test_actor()
        total_data_size_gb = (
            self.config.tensor_count * self.config.tensor_size_kb / (1000 * 1000)
        )  # Convert KB to GB

        # warm up
        for i in range(self.config.warmup_times):
            ray.get(sender_actor.generate_tensor.remote())
            data_ref = sender_actor.transport_tensor_via_yr.remote()
            ray.get(receiver_actor.recv_tensor.remote(data_ref))

        # Run actual test for count iterations
        transport_times = []
        logger.info(f"Starting transport operation ({self.config.count} iterations)...")
        for iteration in range(self.config.count):
            ray.get(sender_actor.generate_tensor.remote())
            start_transport = time.perf_counter()
            data_ref = sender_actor.transport_tensor_via_yr.remote()
            ray.get(receiver_actor.recv_tensor.remote(data_ref))
            end_transport = time.perf_counter()
            transport_time = end_transport - start_transport
            transport_times.append(transport_time)
            logger.info(
                f"Iteration {iteration + 1}/{self.config.count}: {transport_time:.8f}s"
            )

        # Calculate statistics
        latency_percentiles = self.calculate_latency_percentiles(transport_times)
        throughput_stats = self.calculate_throughput(
            total_data_size_gb, transport_times
        )

        # Log performance summary
        mode_name = f"{self.config.backend.upper()} {self.init_mode.upper()} {self.remote_mode.upper()}"
        self.log_performance_summary(
            logger=logger,
            test_name=mode_name,
            total_data_size_gb=total_data_size_gb,
            iterations=self.config.count,
            latency_percentiles=latency_percentiles,
            throughput_stats=throughput_stats,
        )

    def close(self):
        """Cleanup resources via cleanup_yr_resources and optionally etcd."""
        logger.info(f"Cleaning up YR {self.init_mode} resources...")
        try:
            cleanup_yr_resources()
            logger.info(f"YR {self.init_mode} resources cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup YR resources: {e}")

        # Cleanup auto-started etcd if applicable (CI scenario)
        if self.etcd_util:
            self.etcd_util.close()


def main():
    config = parse_args()

    if config.backend == "yr":
        tester = YRDirectTransportTester(
            config,
            init_mode=config.init_mode,
            remote_mode=config.placement,
        )
    elif config.backend == "hccl":
        raise NotImplementedError("HCCL transport test not implemented yet")
    else:
        raise ValueError(f"Unsupported backend: {config.backend}")

    try:
        tester.run_test()
    finally:
        tester.close()


if __name__ == "__main__":
    main()
