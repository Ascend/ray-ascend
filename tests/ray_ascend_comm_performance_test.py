import argparse
import logging
import sys
import time
from pathlib import Path

import ray
import torch
from omegaconf import OmegaConf
from ray.experimental import register_tensor_transport

from ray_ascend.direct_transport import YRTensorTransport

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
    def decorator(func):
        def wrapper(*args, **kwargs):
            pass

        return wrapper

    return decorator


@ray.remote
class WorkerActor:
    def __init__(self, config):
        check_npu_is_available()
        self.config = config
        self.data = None

    @ray.method
    def setup_yr_ds(self):
        pass

    def generate_tensor(self) -> torch.Tensor:
        self.data = torch.randn(
            self.config.global_batch_size,
            self.config.seq_length,
            device=self.config.device,
        )
        return self.data

    @ray.method(tensor_transport="YR")
    def transport_tensor_via_yr(self, data) -> torch.Tensor:
        return self.data

    @ray.method(tensor_transport="HCCL")
    def transport_tensor_via_hccl(self, data) -> torch.Tensor:
        return self.data

    def recv_tensor(self, data_ref) -> torch.Tensor:
        if not isinstance(data_ref, ray.ObjectRef):
            raise ValueError("Expected a Ray ObjectRef for the tensor data.")
        data = ray.get(data_ref)
        logger.info(f"Received tensor of size {data.shape}")
        return data


class HCCLTransportBandwidthTester:
    pass


class YRDirectTransportBandwidthTester:
    def __init__(self, config, remote_mode=False):
        self.config = config
        self.remote_mode = remote_mode
        self._initialize_data_system()

    def _initialize_data_system(self):
        if self.remote_mode == "remote":
            logger.info("Initializing data system client in remote mode...")
            self.writer_actor = WorkerActor.options(
                resources={f"node:{HEAD_NODE_IP}": 0.001}
            ).remote(self.config)
            ray.get(self.writer_actor.setup_yr_ds.remote())
            self.reader_actor = WorkerActor.options(
                resources={f"node:{WORKER_NODE_IP}": 0.001}
            ).remote(self.config)
            ray.get(self.reader_actor.setup_yr_ds.remote())
        else:
            logger.info("Initializing data system client in local mode...")
            self.writer_actor = WorkerActor.remote(self.config)
            ray.get(self.writer_actor.setup_yr_ds.remote())
            self.reader_actor = WorkerActor.remote(self.config)
            ray.get(self.reader_actor.setup_yr_ds.remote())

    def run_bandwidth_test(self):
        total_data_size_gb = compute_total_size(
            batch_size=self.config.global_batch_size,
            seq_length=self.config.seq_length,
            device=self.config.device,
        )
        logger.info("Creating large batch for bandwidth test...")
        start_create_data = time.time()
        ray.get(self.writer_actor.generate_tensor.remote())
        end_create_data = time.time()
        logger.info(f"Data creation time: {end_create_data - start_create_data:.8f}s")

        logger.info("Starting transport operation...")
        start_transport = time.time()
        data_ref = self.writer_actor.transport_tensor_via_yr.remote()
        results = ray.get(self.reader_actor.recv_tensor.remote(data_ref))
        assert torch.equal(
            results, self.writer_actor.data
        ), "Data mismatch after transport!"
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
