import logging
import os
import random
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Optional

import pytest
import requests

logger = logging.getLogger(__name__)


def check_dscli_available() -> bool:
    return shutil.which("dscli") is not None


def get_free_port():
    """Find and return an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def check_etcd_installed() -> None:
    """Raise RuntimeError if 'etcd' is not found in PATH."""
    if shutil.which("etcd") is None:
        raise RuntimeError(
            "'etcd' is not installed or not found in PATH. Please install etcd and ensure it's accessible from the command line."
        )


def start_etcd(
    host: str = "127.0.0.1",
    client_port: Optional[int] = None,
    peer_port: Optional[int] = None,
    max_retries: int = 3,
):
    """Start etcd in a subprocess and wait until it's healthy."""
    check_etcd_installed()

    for attempt in range(max_retries):
        etcd_data_dir = tempfile.mkdtemp(prefix=f"etcd-data-{int(time.time())}")

        client_port_ = client_port if client_port is not None else get_free_port()
        peer_port_ = peer_port if peer_port is not None else get_free_port()

        client_addr = f"http://{host}:{client_port_}"
        peer_addr = f"http://{host}:{peer_port_}"
        unique_name = f"etcd-{client_addr}"
        cmd = [
            "etcd",
            "--name",
            unique_name,
            "--data-dir",
            etcd_data_dir,
            "--listen-client-urls",
            client_addr,
            "--advertise-client-urls",
            client_addr,
            "--listen-peer-urls",
            peer_addr,
            "--initial-advertise-peer-urls",
            peer_addr,
            "--initial-cluster",
            f"{unique_name}={peer_addr}",
        ]

        etcd_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),
        )

        # Wait for etcd to become ready (max ~3 seconds)
        for _ in range(10):
            try:
                resp = requests.get(f"{client_addr}/health", timeout=1)
                is_etcd_healthy: bool = (
                    resp.status_code == requests.codes.ok
                    and resp.json().get("health") == "true"
                )
                if is_etcd_healthy:
                    logger.info(
                        f"etcd started on client={client_addr}, peer={peer_addr}"
                    )
                    etcd_addr = client_addr.replace("http://", "")
                    return etcd_addr, etcd_proc, etcd_data_dir
            except requests.RequestException:
                pass
            time.sleep(0.3)
        else:
            # Cleanup failed process before retry
            etcd_proc.terminate()
            etcd_proc.wait(timeout=5)

            # delete outdated temp etcd directory
            if os.path.exists(etcd_data_dir):
                shutil.rmtree(etcd_data_dir, ignore_errors=True)

        # Small randomized backoff before retry
        if attempt + 1 < max_retries:
            time.sleep(0.1 + random.uniform(0, 0.2))

    raise RuntimeError(f"Failed to start etcd after {max_retries} retries")


def start_datasystem(
    etcd_addr: str,
    worker_host: str = "127.0.0.1",
    worker_port: Optional[int] = None,
    max_retries: int = 3,
):
    """Start yuanrong datasystem worker via dscli and verify success by checking '[  OK  ]' in output."""
    for attempt in range(max_retries):

        worker_port_ = worker_port if worker_port is not None else get_free_port()
        cmd = [
            "dscli",
            "start",
            "-w",
            "--worker_address",
            f"{worker_host}:{worker_port_}",
            "--etcd_address",
            etcd_addr,
        ]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=90,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("dscli start timed out")

        if result.returncode == 0 and "[  OK  ]" in result.stdout:
            return worker_host, worker_port_

        if attempt == max_retries - 1:
            logger.error(f"dscli start failed. Final dscli output:\n{result.stdout}")
        else:
            # Small randomized backoff before retry
            time.sleep(0.1 + random.uniform(0, 0.2))

    raise RuntimeError(f"Failed to start datasystem after {max_retries} retries")
