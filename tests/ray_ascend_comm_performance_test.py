import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf
from ray.experimental import register_tensor_transport

from ray_ascend.direct_transport import YRTensorTransport
from ray_ascend.tests.direct_transport.conftest import (
    start_datasystem,
    start_etcd,
)

parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

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
  global_batch_size: 1024
  seq_length: 8192
  num_global_batch: 1
  num_data_storage_units: 8
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


def yr_is_available_in_actor(actor: ray.ActorHandle) -> bool:
    gpu_object_manager = ray._private.worker.global_worker.gpu_object_manager
    return bool(gpu_object_manager.actor_has_tensor_transport(actor, "YR"))


def compute_total_size(batch_size: int, seq_length: int) -> float:
    total_size_bytes = batch_size * seq_length * 4
    total_size_gb = total_size_bytes / (1024**3)
    logger.info(f"Total data size: {total_size_gb:.6f} GB")

    return total_size_gb


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, help="Backend must be ['yr', 'hccl']")
    parser.add_argument(
        "--placement", type=str, default="local", help="['local', 'remote']"
    )
    parser.add_argument(
        "--transport", type=str, help="['tcp', 'rdma'] for yr or ['hccs'] for hccl"
    )
    parser.add_argument("--device", type=str, default="cpu", help="['npu', 'cpu']")
    args = parser.parse_args()
    return vars(args)


# Todo: use decorator to start etcd and ds
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
        check_npu_is_available()
        self.config = config
        self.node_ip = node_ip
        self.data = None

    def setup_yr_ds(self, etcd_addr: str):
        # If an etcd address is provided, use it; otherwise start a local etcd
        self.worker_host, self.worker_port = (
            start_datasystem(etcd_addr)
            if self.node_ip is None
            else start_datasystem(etcd_addr, host=self.node_ip)
        )
        os.environ["YR_DS_WORKER_HOST"] = self.worker_host
        os.environ["YR_DS_WORKER_PORT"] = str(self.worker_port)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"

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

    @ray.method(tensor_transport="HCCL")
    def transport_tensor_via_hccl(self) -> torch.Tensor:
        return self.data

    def recv_tensor(self, data_ref) -> torch.Tensor:
        if not isinstance(data_ref, ray.ObjectRef):
            raise ValueError("Expected a Ray ObjectRef for the tensor data.")
        data = ray.get(data_ref)
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
        if self.remote_mode == "remote":
            logger.info("Initializing data system client in remote mode...")
            # etcd is started by the class decorator; pass its address to actors
            # TODO: support npu resource allocation for actors
            self.writer_actor = WorkerActor.options(
                resources={f"node:{HEAD_NODE_IP}": 0.001}
            ).remote(self.config, HEAD_NODE_IP)
            ray.get(
                self.writer_actor.setup_yr_ds.remote(
                    getattr(self, "_etcd_addr", None), None
                )
            )
            self.reader_actor = WorkerActor.options(
                resources={f"node:{WORKER_NODE_IP}": 0.001}
            ).remote(self.config, WORKER_NODE_IP)
            ray.get(
                self.reader_actor.setup_yr_ds.remote(
                    getattr(self, "_etcd_addr", None), None
                )
            )
        else:
            logger.info("Initializing data system client in local mode...")
            self.writer_actor = WorkerActor.remote(self.config)
            ray.get(
                self.writer_actor.setup_yr_ds.remote(
                    getattr(self, "_etcd_addr", None), None
                )
            )
            self.reader_actor = WorkerActor.remote(self.config)
            ray.get(
                self.reader_actor.setup_yr_ds.remote(
                    getattr(self, "_etcd_addr", None), None
                )
            )

    def run_bandwidth_test(self):
        total_data_size_gb = compute_total_size(
            batch_size=self.config.global_batch_size,
            seq_length=self.config.seq_length,
        )
        logger.info("Creating large batch for bandwidth test...")
        start_create_data = time.time()
        data = ray.get(self.writer_actor.generate_tensor.remote())
        end_create_data = time.time()
        logger.info(f"Data creation time: {end_create_data - start_create_data:.8f}s")

        assert yr_is_available_in_actor(
            self.writer_actor
        ), "YR transport is not available in writer actor!"
        assert yr_is_available_in_actor(
            self.reader_actor
        ), "YR transport is not available in reader actor!"

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

        mode_name = "TQ REMOTE" if self.remote_mode else "TQ NORMAL"
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
    logger.info(f"Test configuration: {OmegaConf.to_yaml(config)}")
    config = OmegaConf.merge(data_conf, config)
    if config.device == "npu":
        check_npu_is_available()
    if config.backend == "yr":
        tester = YRDirectTransportBandwidthTester(config, remote_mode=config.placement)
    else:
        raise NotImplementedError(f"Unsupported backend: {config.backend}")

    tester.run_bandwidth_test()


if __name__ == "__main__":
    main()
