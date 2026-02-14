import argparse
import logging
import os
import shutil
import subprocess
import time
from typing import Optional

import ray
import torch
from direct_transport.conftest import (
    start_datasystem,
    start_etcd,
)
from omegaconf import OmegaConf
from ray.experimental import register_tensor_transport

from ray_ascend.direct_transport import YRTensorTransport

register_tensor_transport("YR", ["npu", "cpu"], YRTensorTransport)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

########################################################################
# Please set up Ray cluster before running this script
########################################################################
HEAD_NODE_IP = "NodeA"  # Replace with your head node IP
WORKER_NODE_IP = "NodeB"  # Replace with your worker node IP


# This is the Medium setting of the performance test.
# You can modify the parameters according to
# https://www.yuque.com/haomingzi-lfse7/lhp4el/tml8ke0zkgn6roey?singleDoc#
config_str = """
  tensor_size: 1024
"""
data_conf = OmegaConf.create(config_str)


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


# TODO: support more configurations (Currently only YR with NPU is supported) and config file parsing
def parse_args() -> dict:
    """
    The following parameters are not currently supported:
    transport
    warmup-times
    output-format
    config-file
    """
    arg_configs = [
        {
            "name": "--backend",
            "type": str,
            "choices": ["yr", "hccl"],
            "required": True,
            "help": "Transport backend: 'yr' for YR Direct Transport, 'hccl' for HCCL."
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
            )
        },
        {
            "name": "--transport",
            "type": str,
            "choices": ["tcp", "rdma", "hccs"],
            "help": "Transport protocol: 'tcp' or 'rdma' for 'yr'; 'hccs' for 'hccl'."
        },
        {
            "name": "--device",
            "type": str,
            "choices": ["npu", "cpu"],
            "default": "cpu",
            "help": "Device to run tensors on: 'npu' or 'cpu'."
        },
        {
            "name": "--head-node-ip",
            "type": str,
            "help": "IP address of the Ray head node. Required in 'remote' mode; driver must run on head node."
        },
        {
            "name": "--worker-node-ip",
            "type": str,
            "help": "IP address of the worker node. Required in 'remote' mode."
        },
        {
            "name": "--tensor-size",
            "type": int,
            "default": 1024,
            "help": "Total number of elements in the tensor to transport (default: 1024)."
        },
        {
            "name": "--output-format",
            "type": str,
            "choices": ["stdout", "json", "csv"],
            "default": "stdout",
            "help": "Output format for performance results (default: stdout)."
        },
        {
            "name": "--warmup-times",
            "type": int,
            "default": 3,
            "help": "Number of warmup iterations before measurement (default: 3)."
        },
        {
            "name": "--config-file",
            "type": str,
            "help": (
                "Path to a YAML config file with test parameters. "
                "Command-line arguments override config file settings."
            )
        },
    ]

    
    parser = argparse.ArgumentParser()
    for arg in arg_configs:
        parser.add_argument(arg["name"], **{k: v for k, v in arg.items() if k != "name"})
    args = parser.parse_args()
    config = vars(args)
    if config["placement"] == "remote":
    if not config.get("head_node_ip"):
        parser.error("--head-node-ip is required when --placement=remote")
    if not config.get("worker_node_ip"):
        parser.error("--worker-node-ip is required when --placement=remote")
    logger.info(f"Test configuration:\n{OmegaConf.to_yaml(config)}")
    return config


def decorate_with_transport(transport_name):
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
class WorkerActor:
    def __init__(self, config, node_ip=None):
        self.config = config
        self.node_ip = node_ip
        self.data = None
        # TODO: enhance robustness of device setting
        torch.npu.set_device(0)

    def setup_yr_ds(self, etcd_addr: str, *, worker_host: Optional[str] = None, connect_only_info: Optional[tuple] = None):
        # If an etcd address is provided, use it; otherwise start a local etcd
        self.worker_host = self.worker_port = None
        if connect_only_info is None:
            print(f"Starting datasystem with etcd at: {etcd_addr}")
            self.worker_host, self.worker_port = (
                start_datasystem(etcd_addr)
                if self.node_ip is None
                else start_datasystem(etcd_addr, worker_host=self.node_ip)
            )
        else:
            print(f"Starting datasystem with connect_only_info: {connect_only_info}")
            self.worker_host, self.worker_port = connect_only_info

        os.environ["YR_DS_WORKER_HOST"] = self.worker_host
        os.environ["YR_DS_WORKER_PORT"] = str(self.worker_port)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
        return self.worker_host, self.worker_port

    def close_yr_ds(self):
        try:
            ds_stop_cmd = [
                "dscli",
                "stop",
                "--worker_address",
                f"{self.worker_host}:{self.worker_port}",
            ]
            subprocess.run(ds_stop_cmd, check=True, timeout=180)

        except Exception as e:
            logger.error(f"Failed to stop datasystem: {e}")

    def generate_tensor(self) -> torch.Tensor:
        self.data = torch.randn(
            self.config.global_batch_size,
            self.config.seq_length,
            device=self.config.device,
        )
        return self.data

    @ray.method(tensor_transport="YR")
    def transport_tensor_via_yr(self) -> torch.Tensor:
        return self.data

    # TODO(dpj): Add HCCL transport test after HCCL transport is supported in Ray-ascend.
    # @ray.method(tensor_transport="HCCL")
    # def transport_tensor_via_hccl(self) -> torch.Tensor:
    #     return self.data

    def recv_tensor(self, data: torch.Tensor) -> torch.Tensor:
        logger.info(f"Received tensor of size {data.shape}")
        return data


class HCCLTransportBandwidthTester:
    pass


@decorate_with_transport("YR")
class YRDirectTransportBandwidthTester:
    def __init__(self, config, remote_mode=False):
        self.config = config
        self.remote_mode = remote_mode
        self._initialize_data_system()

    def _initialize_data_system(self):
        # TODO: support cpu transport test after YR transport supports cpu tensors
        if self.remote_mode == "remote":
            logger.info("Initializing data system client in remote mode...")
            # etcd is started by the class decorator; pass its address to actors
            self.writer_actor = WorkerActor.options(
                resources={f"node:{HEAD_NODE_IP}": 0.001, "NPU": 1}
            ).remote(self.config, HEAD_NODE_IP)
            ray.get(
                self.writer_actor.setup_yr_ds.remote(getattr(self, "_etcd_addr", None), worker_host=HEAD_NODE_IP)
            )
            self.reader_actor = WorkerActor.options(
                resources={f"node:{WORKER_NODE_IP}": 0.001, "NPU": 1}
            ).remote(self.config, WORKER_NODE_IP)
            ray.get(
                self.reader_actor.setup_yr_ds.remote(getattr(self, "_etcd_addr", None), worker_host=WORKER_NODE_IP)
            )
        else:
            logger.info("Initializing data system client in local mode...")
            logger.info(f"etcd address is {getattr(self, '_etcd_addr', None)}")
            self.writer_actor = WorkerActor.options(resources={"NPU": 1}).remote(
                self.config
            )
            local_ds_info = ray.get(
                self.writer_actor.setup_yr_ds.remote(getattr(self, "_etcd_addr", None))
            )
            self.reader_actor = WorkerActor.options(resources={"NPU": 1}).remote(
                self.config
            )
            ray.get(
                self.reader_actor.setup_yr_ds.remote(
                    getattr(self, "_etcd_addr", None), local_ds_info
                )
            )

    def run_bandwidth_test(self):
        total_data_size_gb = compute_total_size(self.config.tensor_size)
        logger.info("Creating large batch for bandwidth test...")
        start_create_data = time.time()
        data = ray.get(self.writer_actor.generate_tensor.remote())
        end_create_data = time.time()
        logger.info(f"Data creation time: {end_create_data - start_create_data:.8f}s")

        # warm up
        data_ref = self.writer_actor.transport_tensor_via_yr.remote()
        results = ray.get(self.reader_actor.recv_tensor.remote(data_ref))

        logger.info("Starting transport operation...")
        start_transport = time.time()
        data_ref = self.writer_actor.transport_tensor_via_yr.remote()
        results = ray.get(self.reader_actor.recv_tensor.remote(data_ref))
        assert torch.equal(results, data), "Data mismatch after transport!"
        end_transport = time.time()
        transport_time = end_transport - start_transport

        transport_throughput_gbps = (total_data_size_gb * 8) / transport_time
        logger.info(f"transport cost time: {transport_time:.8f}s")
        logger.info(f"Transport Throughput: {transport_throughput_gbps:.8f} Gb/s")
        time.sleep(2)

        mode_name = "YR REMOTE" if self.remote_mode else "YR NORMAL"
        logger.info("=" * 60)
        logger.info(f"{mode_name} BANDWIDTH TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Data Size: {total_data_size_gb:.6f} GB")
        logger.info(f"Transport Time: {transport_time:.8f}s")
        logger.info(f"Transport Throughput: {transport_throughput_gbps:.8f} Gb/s")
        logger.info(
            f"Network Round-trip Throughput: {(total_data_size_gb * 8) / transport_time:.8f} Gb/s"
        )

        w = self.writer_actor.close_yr_ds.remote()
        r = self.reader_actor.close_yr_ds.remote()
        ray.get([w, r])


def main():
    config = parse_args()
    config = OmegaConf.merge(data_conf, config)

    # TODO: support for remote actor to check NPU device
    if config.device == "npu":
        check_npu_is_available()

    if config.backend == "yr":
        tester = YRDirectTransportBandwidthTester(config, remote_mode=config.placement)
    else:
        raise NotImplementedError(f"Unsupported backend: {config.backend}")

    tester.run_bandwidth_test()


if __name__ == "__main__":
    main()
