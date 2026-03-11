import argparse
import logging
import os
import shutil
import subprocess
import time
from typing import Optional

import ray
import torch
import yaml
from omegaconf import OmegaConf
from ray.experimental import register_tensor_transport

from ray_ascend.direct_transport import YRTensorTransport
from ray_ascend.utils import (
    start_datasystem,
    start_etcd,
)

register_tensor_transport("YR", ["npu", "cpu"], YRTensorTransport)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


HEAD_NODE_IP = "NodeA"
WORKER_NODE_IP = "NodeB"


# This is the Medium setting of the performance test.
# You can modify the parameters according to
# https://www.yuque.com/haomingzi-lfse7/lhp4el/tml8ke0zkgn6roey?singleDoc#


def check_npu_is_available() -> None:
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


def compute_total_size(tensor_size: int) -> float:
    total_size_bytes = tensor_size * 4  # Assuming float32 (4 bytes per element)
    total_size_gb = total_size_bytes / (1024**3)
    logger.info(f"Total data size: {total_size_gb:.6f} GB")

    return total_size_gb


def load_config_from_file(config_file: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config or {}


# TODO: support more configurations (Currently only YR with NPU is supported) and config file parsing
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
            "name": "--transport",
            "type": str,
            "choices": ["tcp", "rdma", "hccs"],
            "help": "Transport protocol: 'tcp' or 'rdma' for 'yr'; 'hccs' for 'hccl'.",
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
            "name": "--tensor-size",
            "type": int,
            "default": 1024,
            "help": "Total number of elements in the tensor to transport (default: 1024).",
        },
        {
            "name": "--output-format",
            "type": str,
            "choices": ["stdout", "json", "csv"],
            "default": "stdout",
            "help": "Output format for performance results (default: stdout).",
        },
        {
            "name": "--warmup-times",
            "type": int,
            "default": 3,
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

    logger.info(f"Test configuration:\n{OmegaConf.to_yaml(vars(final_args))}")
    return final_args


def decorate_with_transport():
    def decorator(cls):
        # Class decorator: start a single etcd for the tester instance
        orig_init = cls.__init__

        def __init__(self, *args, **kwargs):
            # Determine remote_mode from args/kwargs (orig_init signature: (self, config, remote_mode=False))
            remote_mode = kwargs.get("remote_mode", False)
            if not remote_mode and len(args) >= 2:
                remote_mode = args[1]

            etcd_host = HEAD_NODE_IP if remote_mode == "remote" else None

            # Start etcd before running original init so _initialize_data_system can use _etcd_addr
            if etcd_host is not None:
                etcd_addr, etcd_proc, etcd_data_dir = start_etcd(host=etcd_host)
            else:
                etcd_addr, etcd_proc, etcd_data_dir = start_etcd()

            self._etcd_addr = etcd_addr
            self._etcd_proc = etcd_proc
            self._etcd_data_dir = etcd_data_dir

            # Now call original initializer which may create actors and use _etcd_addr
            orig_init(self, *args, **kwargs)

        def _close_etcd(self):
            if hasattr(self, "_etcd_proc") and self._etcd_proc:
                try:
                    self._etcd_proc.terminate()
                    self._etcd_proc.wait(timeout=5)
                except Exception:
                    pass
            if hasattr(self, "_etcd_data_dir") and self._etcd_data_dir:
                try:
                    shutil.rmtree(self._etcd_data_dir, ignore_errors=True)
                except Exception:
                    pass

        def __del__(self):
            try:
                _close_etcd(self)
            except Exception:
                pass

        cls.__init__ = __init__
        cls._close_etcd = _close_etcd
        cls.__del__ = __del__
        return cls

    return decorator


@ray.remote
class DataSystemActor:

    def __init__(self, etcd_addr: str, node_ip: Optional[str] = None):
        self.etcd_addr = etcd_addr
        self.node_ip = node_ip
        self.worker_host: Optional[str] = None
        self.worker_port: Optional[int] = None
        self.ds_started = False

    def start_datasystem(self):
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

    def stop_datasystem(self) -> None:
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
class TensorTransportActor:

    def __init__(self, config, node_ip: Optional[str] = None):
        self.config = config
        self.node_ip = node_ip
        self.data: Optional[torch.Tensor] = None
        self.ds_info: Optional[tuple[Optional[str], Optional[int]]] = None
        host = os.getenv("YR_DS_WORKER_HOST")
        port_str = os.getenv("YR_DS_WORKER_PORT")

        if host and port_str:
            self.ds_info = (host, int(port_str))
            logger.info(f"DataSystem info loaded from environment: {self.ds_info}")
        # TODO: enhance robustness of device setting
        torch.npu.set_device(0)

    def setup_yr_env(self, ds_info: tuple[str, int]) -> None:
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
        self.data = torch.randn(
            self.config.tensor_size,
            device=self.config.device,
        )
        logger.info(f"Generated tensor of shape {self.data.shape}")
        return self.data

    @ray.method(tensor_transport="YR")
    def transport_tensor_via_yr(self) -> torch.Tensor:
        if self.data is None:
            raise RuntimeError("Tensor not generated yet. Call generate_tensor first.")
        return self.data

    def recv_tensor(self, data: torch.Tensor) -> torch.Tensor:
        logger.info(f"Received tensor of shape {data.shape}")
        return data


class HCCLTransportBandwidthTester:
    pass


@decorate_with_transport()
class YRDirectTransportBandwidthTester:
    def __init__(self, config, remote_mode="local"):
        self.config = config
        self.remote_mode = remote_mode
        self.head_ds_actor: Optional[ray.actor.ActorHandle] = None
        self.worker_ds_actor: Optional[ray.actor.ActorHandle] = None
        self._initialize_data_system()

    def _initialize_data_system(self):
        if self.remote_mode == "remote":
            logger.info("Initializing data system client in remote mode...")
            # etcd is started by the class decorator; pass its address to actors
            self.head_ds_actor = DataSystemActor.options(
                resources={f"node:{HEAD_NODE_IP}": 0.001, "NPU": 1}
            ).remote(getattr(self, "_etcd_addr", None), node_ip=HEAD_NODE_IP)
            self.head_ds_info = ray.get(self.head_ds_actor.start_datasystem.remote())
            self.worker_ds_actor = DataSystemActor.options(
                resources={f"node:{WORKER_NODE_IP}": 0.001, "NPU": 1}
            ).remote(getattr(self, "_etcd_addr", None), node_ip=WORKER_NODE_IP)
            self.worker_ds_info = ray.get(
                self.worker_ds_actor.start_datasystem.remote()
            )
        else:
            logger.info("Initializing data system client in local mode...")
            logger.info(f"etcd address is {getattr(self, '_etcd_addr', None)}")
            self.head_ds_actor = self.worker_ds_actor = DataSystemActor.options(
                resources={"NPU": 1}
            ).remote(getattr(self, "_etcd_addr", None))
            self.head_ds_info = self.worker_ds_info = ray.get(
                self.head_ds_actor.start_datasystem.remote()
            )

    def _initialize_test_actor(self):
        # TODO: support cpu transport test after YR transport supports cpu tensors
        if self.remote_mode == "remote":
            logger.info("Initializing data system client in remote mode...")
            # etcd is started by the class decorator; pass its address to actors
            writer_actor = TensorTransportActor.options(
                resources={f"node:{HEAD_NODE_IP}": 0.001, "NPU": 1}
            ).remote(self.config, HEAD_NODE_IP)

            reader_actor = TensorTransportActor.options(
                resources={f"node:{WORKER_NODE_IP}": 0.001, "NPU": 1}
            ).remote(self.config, WORKER_NODE_IP)

        else:
            logger.info("Initializing data system client in local mode...")
            logger.info(f"etcd address is {getattr(self, '_etcd_addr', None)}")
            writer_actor = TensorTransportActor.options(resources={"NPU": 1}).remote(
                self.config
            )
            reader_actor = TensorTransportActor.options(resources={"NPU": 1}).remote(
                self.config
            )

        ray.get(reader_actor.setup_yr_env.remote(self.worker_ds_info))
        ray.get(writer_actor.setup_yr_env.remote(self.head_ds_info))
        return writer_actor, reader_actor

    def run_bandwidth_test(self):
        writer_actor, reader_actor = self._initialize_test_actor()
        total_data_size_gb = compute_total_size(self.config.tensor_size)
        logger.info("Creating large batch for bandwidth test...")
        start_create_data = time.time()
        data = ray.get(writer_actor.generate_tensor.remote())
        end_create_data = time.time()
        logger.info(f"Data creation time: {end_create_data - start_create_data:.8f}s")

        # warm up
        for i in range(self.config.warmup_times):
            data_ref = writer_actor.transport_tensor_via_yr.remote()
            results = ray.get(reader_actor.recv_tensor.remote(data_ref))

        logger.info("Starting transport operation...")
        start_transport = time.time()
        data_ref = writer_actor.transport_tensor_via_yr.remote()
        results = ray.get(reader_actor.recv_tensor.remote(data_ref))
        assert torch.equal(results, data), "Data mismatch after transport!"
        end_transport = time.time()
        transport_time = end_transport - start_transport

        transport_throughput_gbps = (total_data_size_gb * 8) / transport_time
        logger.info(f"transport cost time: {transport_time:.8f}s")
        logger.info(f"Transport Throughput: {transport_throughput_gbps:.8f} Gb/s")
        time.sleep(2)

        mode_name = f"{self.config.backend.upper()} {self.remote_mode.upper()}"
        logger.info("=" * 60)
        logger.info(f"{mode_name} BANDWIDTH TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Data Size: {total_data_size_gb:.6f} GB")
        logger.info(f"Transport Time: {transport_time:.8f}s")
        logger.info(f"Transport Throughput: {transport_throughput_gbps:.8f} Gb/s")
        logger.info(
            f"Network Round-trip Throughput: {(total_data_size_gb * 8) / transport_time:.8f} Gb/s"
        )

    def close_datasystem(self):
        if self.head_ds_actor:
            ray.get(self.head_ds_actor.stop_datasystem.remote())
        if self.worker_ds_actor and self.worker_ds_actor != self.head_ds_actor:
            ray.get(self.worker_ds_actor.stop_datasystem.remote())


def main():
    config = parse_args()

    # TODO: support for remote actor to check NPU device
    if config.device == "npu":
        check_npu_is_available()

    if config.backend == "yr":
        tester = YRDirectTransportBandwidthTester(config, remote_mode=config.placement)
    elif config.backend == "hccl":
        raise NotImplementedError("HCCL transport test not implemented yet")
    else:
        raise ValueError(f"Unsupported backend: {config.backend}")

    try:
        tester.run_bandwidth_test()
    finally:
        if config.backend == "yr":
            tester.close_datasystem()


if __name__ == "__main__":
    main()
