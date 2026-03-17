import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import ray
import torch
import yaml
from ray.experimental import register_tensor_transport

from ray_ascend.direct_transport import YRTensorTransport
from ray_ascend.utils import (
    start_datasystem,
    start_etcd,
)

register_tensor_transport("YR", ["npu", "cpu"], YRTensorTransport)

# Add parent directory to sys.path for importing base_perftest
sys.path.insert(0, str(Path(__file__).parent))

from base_perftest import RayAscendBaseTester

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


HEAD_NODE_IP = "NodeA"
WORKER_NODE_IP = "NodeB"


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


def yr_is_available_in_actor(actor: "ray.actor.ActorHandle") -> bool:
    gpu_object_manager = ray._private.worker.global_worker.gpu_object_manager
    return bool(gpu_object_manager.actor_has_tensor_transport(actor, "YR"))


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
            "name": "--tensor-size-kb",
            "type": int,
            "default": 1024,
            "help": "Total number of elements in the tensor to transport (default: 1024).",
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

    logger.info(
        f"Test configuration:\n{yaml.dump(vars(final_args), default_flow_style=False)}"
    )
    return final_args


class EtcdUtil:
    """Utility class to manage etcd process and lifecycle."""

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


@ray.remote
class DataSystemActor:

    def __init__(self, etcd_addr: str, node_ip: Optional[str] = None):
        self.etcd_addr = etcd_addr
        self.node_ip = node_ip
        self.worker_host: Optional[str] = None
        self.worker_port: Optional[int] = None
        self.ds_started = False

    def start_datasystem(self) -> tuple[Optional[str], Optional[int]]:
        if self.ds_started:
            logger.warning("DataSystem already started")
            return self.worker_host, self.worker_port

        try:
            if self.node_ip is None:
                self.worker_host, self.worker_port = start_datasystem(self.etcd_addr)
            else:
                self.worker_host, self.worker_port = start_datasystem(
                    self.etcd_addr, worker_host=self.node_ip
                )

            self.ds_started = True
            logger.info(f"DataSystem started at {self.worker_host}:{self.worker_port}")
            return self.worker_host, self.worker_port
        except Exception as e:
            logger.error(f"Failed to start datasystem: {e}")
            raise

    def stop_datasystem(self):
        if not self.ds_started:
            logger.warning("DataSystem not started, skipping stop")
            return

        try:
            ds_stop_cmd = [
                "dscli",
                "stop",
                "--worker_address",
                f"{self.worker_host}:{self.worker_port}",
            ]
            subprocess.run(ds_stop_cmd, check=True, timeout=90)
            self.ds_started = False
            logger.info("DataSystem stopped successfully")
        except Exception as e:
            logger.error(f"Failed to stop datasystem: {e}")

    def get_datasystem_info(self) -> tuple[Optional[str], Optional[int]]:
        if not self.ds_started:
            raise RuntimeError("DataSystem not started yet")
        return self.worker_host, self.worker_port


@ray.remote
class YRTensorTransportActor:

    def __init__(self, config: argparse.Namespace, node_ip: Optional[str] = None):
        register_tensor_transport("YR", ["npu", "cpu"], YRTensorTransport)
        self.config = config
        self.node_ip = node_ip
        self.data: Optional[torch.Tensor] = None
        self.ds_info: Optional[tuple[Optional[str], Optional[int]]] = None

        # if host and port are provided via environment variables, we can skip starting datasystem automatically.
        # TODO: But etcd will be still started by the test class.
        host = os.getenv("YR_DS_WORKER_HOST")
        port_str = os.getenv("YR_DS_WORKER_PORT")

        if host and port_str:
            self.ds_info = (host, int(port_str))
            logger.info(f"DataSystem info loaded from environment: {self.ds_info}")

        if self.config.device == "npu":
            check_npu_is_available()

    def setup_yr_env(self, ds_info: tuple[str, int]):
        """setup environment variables for YR transport"""
        if self.ds_info:
            logger.warning("DataSystem info already set, skipping environment setup")
            return
        self.ds_info = ds_info
        worker_host, worker_port = ds_info
        os.environ["YR_DS_WORKER_HOST"] = worker_host
        os.environ["YR_DS_WORKER_PORT"] = str(worker_port)
        logger.info(f"DataSystem environment configured: {worker_host}:{worker_port}")

    def generate_tensor(self) -> torch.Tensor:
        # convert KB to number of float32 elements
        seq_len = self.config.tensor_size_kb * 1000 // 4
        self.data = torch.randn(
            seq_len,
            device=self.config.device,
        )
        logger.info(f"Generated tensor of shape {self.data.shape}")
        return self.data

    @ray.method(tensor_transport="YR")
    def transport_tensor_via_yr(self) -> torch.Tensor:
        if self.data is None:
            raise RuntimeError("Tensor not generated yet. Call generate_tensor first.")
        return self.data

    def recv_tensor(self, data: torch.Tensor) -> bool:
        if isinstance(data, torch.Tensor):
            return True
        return False


class HCCLTransportTester:
    pass


class YRDirectTransportTester(RayAscendBaseTester):
    def __init__(self, config: argparse.Namespace, remote_mode: str = "local"):
        self.config = config
        self.remote_mode = remote_mode

        # Initialize etcd
        etcd_host = HEAD_NODE_IP if remote_mode == "remote" else None
        self.etcd_util = EtcdUtil(host=etcd_host) if etcd_host else EtcdUtil()

        self.head_ds_actor: Optional[ray.actor.ActorHandle] = None
        self.worker_ds_actor: Optional[ray.actor.ActorHandle] = None
        self._initialize_data_system()

    def _initialize_data_system(self):
        "Initialize data system workers by DataSystemActor and get their info for later use in test actors."
        if self.remote_mode == "remote":
            logger.info("Initializing data system client in remote mode...")
            # etcd is started by EtcdUtil; pass its address to actors
            self.head_ds_actor = DataSystemActor.options(  # type: ignore[attr-defined]
                resources={f"node:{HEAD_NODE_IP}": 0.001}
            ).remote(self.etcd_util.etcd_addr, node_ip=HEAD_NODE_IP)
            self.head_ds_info = ray.get(self.head_ds_actor.start_datasystem.remote())
            self.worker_ds_actor = DataSystemActor.options(  # type: ignore[attr-defined]
                resources={f"node:{WORKER_NODE_IP}": 0.001}
            ).remote(
                self.etcd_util.etcd_addr, node_ip=WORKER_NODE_IP
            )
            self.worker_ds_info = ray.get(
                self.worker_ds_actor.start_datasystem.remote()
            )
        else:
            logger.info("Initializing data system client in local mode...")
            logger.info(f"etcd address is {self.etcd_util.etcd_addr}")
            self.head_ds_actor = self.worker_ds_actor = DataSystemActor.remote(
                self.etcd_util.etcd_addr
            )
            self.head_ds_info = self.worker_ds_info = ray.get(
                self.head_ds_actor.start_datasystem.remote()
            )

    def _initialize_test_actor(
        self,
    ) -> tuple[ray.actor.ActorHandle, ray.actor.ActorHandle]:
        if self.remote_mode == "remote":
            sender_actor = YRTensorTransportActor.options(  # type: ignore[attr-defined]
                resources={f"node:{HEAD_NODE_IP}": 0.001, "NPU": 1}
            ).remote(self.config, HEAD_NODE_IP)

            receiver_actor = YRTensorTransportActor.options(  # type: ignore[attr-defined]
                resources={f"node:{WORKER_NODE_IP}": 0.001, "NPU": 1}
            ).remote(
                self.config, WORKER_NODE_IP
            )

        else:
            sender_actor = YRTensorTransportActor.options(resources={"NPU": 1}).remote(  # type: ignore[attr-defined]
                self.config
            )
            receiver_actor = YRTensorTransportActor.options(resources={"NPU": 1}).remote(  # type: ignore[attr-defined]
                self.config
            )

        ray.get(receiver_actor.setup_yr_env.remote(self.worker_ds_info))
        ray.get(sender_actor.setup_yr_env.remote(self.head_ds_info))
        return sender_actor, receiver_actor

    def run_test(self):
        sender_actor, receiver_actor = self._initialize_test_actor()
        total_data_size_gb = self.config.tensor_size_kb / (
            1000 * 1000
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
        mode_name = f"{self.config.backend.upper()} {self.remote_mode.upper()}"
        self.log_performance_summary(
            logger=logger,
            test_name=mode_name,
            total_data_size_gb=total_data_size_gb,
            iterations=self.config.count,
            latency_percentiles=latency_percentiles,
            throughput_stats=throughput_stats,
        )

    def close(self):
        """Close datasystem and cleanup etcd."""
        if self.head_ds_actor:
            ray.get(self.head_ds_actor.stop_datasystem.remote())
        if self.worker_ds_actor and self.worker_ds_actor != self.head_ds_actor:
            ray.get(self.worker_ds_actor.stop_datasystem.remote())

        # Clean up etcd
        if hasattr(self, "etcd_util"):
            self.etcd_util.close()


def main():
    config = parse_args()

    if config.backend == "yr":
        tester = YRDirectTransportTester(config, remote_mode=config.placement)
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
