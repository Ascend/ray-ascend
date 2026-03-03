import logging
import os
import random
import shutil
import subprocess
import tempfile
import time
from typing import Optional

import pytest
import requests

try:
    from yr import datasystem

    YR_AVAILABLE = True
except ImportError:
    YR_AVAILABLE = False

from ray_ascend.utils import (
    start_datasystem,
    start_etcd,
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def start_etcd_and_yr():
    """
    Start a temporary etcd service and datasystem worker.
    Automatically shut down after tests.
    Yields (worker_host, worker_port).
    """
    if not YR_AVAILABLE:
        pytest.skip("yr library (yuanrong) not available, skipping YR tests")
    if not check_dscli_available():
        pytest.skip("dscli tool not available, skipping YR tests")

    etcd_proc = etcd_data_dir = None
    worker_host = worker_port = None
    try:
        etcd_addr, etcd_proc, etcd_data_dir = start_etcd()
        worker_host, worker_port = start_datasystem(etcd_addr)
        yield worker_host, worker_port

    finally:
        # Stop datasystem
        if worker_host and worker_port:
            try:
                ds_stop_cmd = [
                    "dscli",
                    "stop",
                    "--worker_address",
                    f"{worker_host}:{worker_port}",
                ]
                subprocess.run(ds_stop_cmd, check=True, timeout=180)
            except Exception as e:
                logger.error(f"Failed to stop datasystem: {e}")

        # Stop etcd
        if etcd_proc:
            etcd_proc.terminate()
            etcd_proc.wait(timeout=5)

        # delete outdated temp etcd directory
        if etcd_data_dir and os.path.exists(etcd_data_dir):
            shutil.rmtree(etcd_data_dir, ignore_errors=True)


if __name__ == "__main__":
    """Debug codes"""
    logging.basicConfig(level=logging.INFO)

    etcd_addr, etcd_proc, etcd_data_dir = start_etcd()
    logger.info(f"etcd's address is: {etcd_addr}")

    worker_host, worker_port = start_datasystem(etcd_addr)
    logger.info(f"Yuanrong datasystem worker's address is: {worker_host}:{worker_port}")

    ds_client = datasystem.KVClient(worker_host, worker_port)
    ds_client.init()
    logger.info("Datasystem client has inited")

    try:
        ds_stop_cmd = [
            "dscli",
            "stop",
            "--worker_address",
            f"{worker_host}:{worker_port}",
        ]
        subprocess.run(ds_stop_cmd, check=True, timeout=180)

    except Exception as e:
        logger.error(f"Failed to stop datasystem: {e}")

    etcd_proc.terminate()
    etcd_proc.wait(timeout=5)

    # delete outdated temp etcd directory
    if etcd_data_dir and os.path.exists(etcd_data_dir):
        shutil.rmtree(etcd_data_dir, ignore_errors=True)
