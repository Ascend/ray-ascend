import logging
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Optional

import ray
import requests

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("RAY_ASCEND_LOGGING_LEVEL", logging.WARNING))


# Reaper script for Parent Process Death Detection cleanup (monitors stdin for EOF, stops ds worker on parent death)
_REAPER_SCRIPT = """
import subprocess
import sys

worker_address = sys.argv[1]

try:
    sys.stdin.read()  # Block until EOF (parent process dies)
except Exception:
    pass

try:
    subprocess.run(
        ["dscli", "stop", "--worker_address", worker_address],
        timeout=90,
        capture_output=True,
    )
except Exception:
    pass
"""


def get_free_port() -> int:
    """Find and return an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def check_etcd_installed():
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
) -> tuple[str, subprocess.Popen, str]:
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


def start_datasystem_worker(
    worker_address: str,
    init_mode: str = "etcd",
    # etcd mode parameters
    etcd_address: Optional[str] = None,
    # metastore mode parameters
    metastore_address: Optional[str] = None,
    is_head: bool = False,
    worker_args: Optional[str] = None,
) -> str:
    """Start YR DS worker (unified function with init_mode parameter).

    Args:
        worker_address: Worker address in format "host:port"
        init_mode: Initialization mode, "etcd" or "metastore"
        etcd_address: Etcd address (required for etcd mode)
        metastore_address: Metastore address (required for metastore mode)
        is_head: Whether this is head node (metastore mode only, starts metastore service)
        worker_args: Additional dscli parameters

    Returns:
        Worker address string

    Raises:
        RuntimeError: Parameter conflict, missing parameter, or startup failure
    """
    if not shutil.which("dscli"):
        raise RuntimeError(
            'dscli executable not found in PATH. Please run `pip install "openyuanrong-datasystem>=0.8.0"`.'
        )

    # Build base command
    cmd = ["dscli", "start", "-w", "--worker_address", worker_address]

    # Branch based on init_mode for mode-specific parameters
    if init_mode == "etcd":
        if not etcd_address:
            raise RuntimeError("etcd mode requires etcd_address")
        cmd.extend(["--etcd_address", etcd_address])
        node_type = "etcd mode"

    elif init_mode == "metastore":
        if not metastore_address:
            raise RuntimeError("metastore mode requires metastore_address")
        cmd.extend(["--metastore_address", metastore_address])
        if is_head:
            cmd.extend(["--start_metastore_service", "true"])
        node_type = "head node" if is_head else "worker node"

    else:
        raise RuntimeError(
            f"Unknown init_mode: {init_mode}. Must be 'etcd' or 'metastore'."
        )

    # Common default parameters for both modes
    cmd.extend(["--arena_per_tenant", "1", "--enable_worker_worker_batch_get", "true"])

    # Append worker_args if provided (common for both modes)
    if worker_args:
        cmd.extend(worker_args.split())

    logger.info(f"Starting YR DS worker ({init_mode}, {node_type}) at {worker_address}")

    # Build environment with ASCEND_RT_VISIBLE_DEVICES if specified (for NPU)
    env = None
    if worker_args:
        device_ids = _parse_remote_h2d_device_ids(worker_args)
        if device_ids:
            env = os.environ.copy()
            env["ASCEND_RT_VISIBLE_DEVICES"] = device_ids
            logger.info(
                f"Setting ASCEND_RT_VISIBLE_DEVICES={device_ids} for dscli subprocess ({node_type} at {worker_address})"
            )

    # Execute command
    ds_result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=90,
        env=env,
        start_new_session=True,
    )

    if ds_result.returncode == 0 and "[  OK  ]" in ds_result.stdout:
        logger.info(
            f"dscli started YR DS worker ({init_mode}, {node_type}) at {worker_address} successfully"
        )
        return worker_address

    raise RuntimeError(
        f"Failed to start YR DS worker ({init_mode}, {node_type}) at {worker_address}. "
        f"Return code: {ds_result.returncode}, Output: {ds_result.stdout}"
    )


def stop_datasystem_worker(worker_address: str) -> None:
    """Stop YR DS worker.

    Args:
        worker_address: Worker address in format "host:port"

    Raises:
        RuntimeError: If dscli stop command fails
    """
    if not shutil.which("dscli"):
        raise RuntimeError(
            'dscli executable not found in PATH. Please run `pip install "openyuanrong-datasystem>=0.8.0"`.'
        )

    try:
        subprocess.run(
            ["dscli", "stop", "--worker_address", worker_address],
            timeout=90,
            capture_output=True,
            check=True,
        )
        logger.info(f"Stopped ds worker at {worker_address}")
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout stopping ds worker at {worker_address}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to stop ds worker at {worker_address}: {e}")


def _parse_remote_h2d_device_ids(worker_args: str) -> Optional[str]:
    """Parse --remote_h2d_device_ids parameter from worker_args string.

    Args:
        worker_args: Worker arguments string, e.g., "--arg1 value1 --remote_h2d_device_ids 0,1,2,3"

    Returns:
        The device IDs string if found and valid, None otherwise.

    Raises:
        RuntimeError: If --remote_h2d_device_ids flag is found but has invalid format.
    """
    if not worker_args:
        return None

    args_list = worker_args.split()

    # Find the index of --remote_h2d_device_ids
    try:
        idx = args_list.index("--remote_h2d_device_ids")
    except ValueError:
        return None

    # Check if there's a value after the flag
    if idx + 1 >= len(args_list):
        raise RuntimeError("--remote_h2d_device_ids flag found but no value provided")

    device_ids = args_list[idx + 1]

    # Validate the format: comma-separated digits
    if not device_ids:
        raise RuntimeError("Empty device IDs value after --remote_h2d_device_ids")

    # Validate each segment is a digit
    parts = device_ids.split(",")
    for part in parts:
        if not part.isdigit():
            raise RuntimeError(
                f"Invalid device ID format: '{device_ids}'. Expected comma-separated digits (e.g., '0,1,2,3')."
            )

    return device_ids


@ray.remote(num_cpus=0.1)
class DataSystemActor:
    """Ray actor to manage YR DS worker on a node.

    Supports both etcd and metastore initialization modes.

    Uses Parent Process Death Detection to ensure ds worker is cleaned up
    when the actor process dies (including when ray stop kills the process).

    Args:
        init_mode: Initialization mode, "etcd" or "metastore"
        etcd_address: Etcd address (required for etcd mode)
        metastore_address: Metastore address (required for metastore mode)
        is_head: Whether this is head node (metastore mode, starts metastore service)
        worker_args: Additional dscli parameters (metastore mode)
        worker_port: Worker port (etcd mode: auto-detect if None)
    """

    def __init__(
        self,
        init_mode: str,
        etcd_address: Optional[str] = None,
        metastore_address: Optional[str] = None,
        is_head: bool = False,
        worker_args: Optional[str] = None,
        worker_port: Optional[int] = None,
    ):
        self.init_mode = init_mode
        self._worker_host: Optional[str] = None
        self._worker_port: Optional[int] = None
        self._worker_address: Optional[str] = None
        self._reaper_process: Optional[subprocess.Popen] = None
        self.etcd_address: Optional[str] = None
        self.metastore_address: Optional[str] = None
        self.is_head: bool = False
        self.worker_args = worker_args or ""

        # Get node IP via Ray API
        self._worker_host = ray.util.get_node_ip_address()

        # Validate and set mode-specific parameters
        if init_mode == "etcd":
            if not etcd_address:
                raise RuntimeError("etcd mode requires etcd_address")
            self.etcd_address = etcd_address
            self._worker_port = worker_port or get_free_port()
        elif init_mode == "metastore":
            if not metastore_address:
                raise RuntimeError("metastore mode requires metastore_address")
            if not worker_port:
                raise RuntimeError("metastore mode requires worker_port")
            self.metastore_address = metastore_address
            self.is_head = is_head
            self._worker_port = worker_port
        else:
            raise RuntimeError(f"Unknown init_mode: {init_mode}")

        self._worker_address = f"{self._worker_host}:{self._worker_port}"
        logger.info(
            f"DataSystemActor initialized ({init_mode}): "
            f"worker_address={self._worker_address}"
        )

    def start(self) -> str:
        """Start the datasystem worker on this node.

        Returns:
            The worker address.

        Raises:
            RuntimeError: If dscli command fails
        """
        assert self._worker_address is not None
        logger.info(f"Starting YR DS worker at {self._worker_address}...")

        worker_address = start_datasystem_worker(
            worker_address=self._worker_address,
            init_mode=self.init_mode,
            etcd_address=self.etcd_address,
            metastore_address=self.metastore_address,
            is_head=self.is_head,
            worker_args=self.worker_args,
        )

        self._worker_host, port_str = worker_address.split(":")
        self._worker_port = int(port_str)
        self._worker_address = worker_address

        # Start reaper process for Parent Process Death Detection cleanup
        self._start_reaper(worker_address)

        logger.info(f"YR DS worker started successfully at {self._worker_address}")
        return worker_address

    def _start_reaper(self, worker_address: str) -> None:
        """Start reaper subprocess for Parent Process Death Detection cleanup."""
        self._reaper_process = subprocess.Popen(
            [sys.executable, "-c", _REAPER_SCRIPT, worker_address],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info(f"Started reaper process for {worker_address}")

    def get_worker_address(self) -> str:
        """Return the worker address."""
        assert self._worker_address is not None
        return self._worker_address

    def get_node_ip(self) -> str:
        """Return the IP address of the node this actor is running on."""
        assert self._worker_host is not None
        return self._worker_host

    def get_metastore_address(self) -> Optional[str]:
        """Return the metastore address (None for etcd mode)."""
        return self.metastore_address

    def stop_worker(self) -> None:
        """Stop the ds worker on this node."""
        if self._worker_address:
            stop_datasystem_worker(self._worker_address)


def cleanup_yr_resources() -> None:
    """Cleanup all YR resources.

    Delegates cleanup to coordinator, then kills the coordinator actor.
    """
    try:
        coordinator = ray.get_actor("YRBackendCoordinator", namespace="yr_backend")
    except ValueError:
        logger.info("YRBackendCoordinator not found, no cleanup needed")
        return

    # Coordinator handles all cleanup (stop workers, remove placement group)
    try:
        ray.get(coordinator.cleanup.remote(), timeout=60)
        logger.info("Coordinator cleanup completed")
    except Exception as e:
        logger.warning(f"Failed to cleanup via coordinator: {e}")

    # Kill coordinator
    try:
        ray.kill(coordinator)
        logger.info("YR resources cleaned up")
    except Exception:
        pass


@ray.remote(
    num_cpus=0.1,
    name="YRBackendCoordinator",
    lifetime="detached",
    namespace="yr_backend",
)
class YRBackendCoordinator:
    """Named actor to coordinate YR backend initialization.

    This actor ensures that YR backend initialization happens only once
    across the Ray cluster. Ray guarantees that a named actor can only
    be created once, avoiding race conditions between multiple processes.

    The coordinator supports both etcd and metastore modes:
    - etcd mode: User provides etcd address, coordinator starts workers on all nodes
    - metastore mode: Coordinator starts metastore service on head node, workers on others

    The coordinator:
    - Creates Placement Group for worker actors
    - Creates DataSystemActor on each node
    - Manages node_worker_addresses mapping
    """

    def __init__(self):
        self._initialized = False
        self._init_mode: Optional[str] = None  # "etcd" or "metastore"
        self._worker_args: str = ""
        self._placement_group = None
        self._etcd_address: Optional[str] = None  # for etcd mode
        self._metastore_address: Optional[str] = None  # for metastore mode
        self._node_worker_addresses = {}  # {node_ip: worker_address}
        self._worker_actors: list = []  # Store actor handles for cleanup

    def _create_placement_group(self, nodes: list):
        """Create placement group with STRICT_SPREAD strategy.

        Args:
            nodes: List of Ray node dicts

        Returns:
            Placement group handle

        Raises:
            RuntimeError: If placement group creation fails or times out
        """
        bundles = [{"CPU": 0.1} for _ in nodes]
        pg = ray.util.placement_group(bundles, strategy="STRICT_SPREAD")

        try:
            ray.get(pg.ready(), timeout=60)
        except ray.exceptions.GetTimeoutError as e:
            try:
                ray.util.remove_placement_group(pg)
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to remove placement group after readiness timeout: {cleanup_error}"
                )
            raise RuntimeError(
                "Timed out waiting for YR placement group to become ready. "
                f"Requested strategy=STRICT_SPREAD, bundles={bundles}. "
                "This may be due to insufficient cluster capacity."
            ) from e
        except Exception as e:
            try:
                ray.util.remove_placement_group(pg)
            except Exception as cleanup_error:
                logger.warning(
                    f"Failed to remove placement group after scheduling failure: {cleanup_error}"
                )
            raise RuntimeError(
                f"Failed to create YR placement group. "
                f"Requested strategy=STRICT_SPREAD, bundles={bundles}."
            ) from e

        logger.info(
            f"Created placement group with {len(bundles)} bundles using STRICT_SPREAD"
        )
        return pg

    def _collect_worker_addresses(self, actors: list) -> None:
        """Collect worker addresses from actors.

        Args:
            actors: List of DataSystemActor handles
        """
        for actor in actors:
            node_ip = ray.get(actor.get_node_ip.remote())
            worker_addr = ray.get(actor.get_worker_address.remote())
            self._node_worker_addresses[node_ip] = worker_addr

    def _get_backend_info_dict(self) -> dict:
        """Return backend info dict."""
        return {
            "init_mode": self._init_mode,
            "worker_args": self._worker_args,
            "etcd_address": self._etcd_address,
            "metastore_address": self._metastore_address,
            "node_worker_addresses": self._node_worker_addresses,
        }

    def stop_all_workers(self) -> None:
        """Stop all ds workers via actor remote calls."""
        if not self._worker_actors:
            logger.info("No worker actors to stop")
            return
        try:
            ray.get(
                [actor.stop_worker.remote() for actor in self._worker_actors],
                timeout=60,
            )
            logger.info("All ds workers stopped")
        except Exception as e:
            logger.warning(f"Failed to stop some workers: {e}")

    def cleanup(self) -> None:
        """Cleanup all resources created by this coordinator.

        Stops ds workers, removes placement group.
        """
        # Stop all ds workers
        self.stop_all_workers()

        # Remove placement group (cleanup remaining actors)
        if self._placement_group:
            try:
                ray.util.remove_placement_group(self._placement_group)
                logger.info("Removed YR placement group")
            except Exception as e:
                logger.warning(f"Failed to remove placement group: {e}")

    def _get_bundle_node_ip(self, pg, bundle_index: int) -> str:
        """Get node IP for a specific bundle in placement group.

        Args:
            pg: Placement group handle
            bundle_index: Bundle index in the placement group

        Returns:
            Node IP address (NodeManagerAddress)

        Raises:
            RuntimeError: If bundle or node not found
        """
        pg_table = ray.util.placement_group_table(pg)
        bundles_to_node = pg_table.get("bundles_to_node_id", {})
        node_id = bundles_to_node.get(bundle_index)
        if not node_id:
            raise RuntimeError(f"Bundle {bundle_index} not found in placement group")

        for node in ray.nodes():
            if node.get("NodeID") == node_id:
                node_ip: str = node.get("NodeManagerAddress", "")
                if node_ip:
                    return node_ip

        raise RuntimeError(f"Node {node_id} not found in cluster")

    def _get_alive_nodes(self) -> list:
        """Get list of alive Ray nodes.

        Returns:
            List of alive node dicts with NodeManagerAddress

        Raises:
            RuntimeError: If no alive nodes found
        """
        nodes = ray.nodes()
        alive_nodes = [n for n in nodes if n.get("Alive", False)]
        if not alive_nodes:
            raise RuntimeError("No alive Ray nodes found")
        return alive_nodes

    def initialize(
        self,
        init_mode: str,
        worker_port: int,
        worker_args: str,
        etcd_address: Optional[str] = None,
        metastore_port: Optional[int] = None,
    ) -> dict:
        """Initialize YR backend.

        All parameters are provided by ensure_yr_backend_initialized.
        Creates Placement Group and DataSystemActor on each node. Can only be called once.
        Subsequent calls with different parameters are ignored - first call wins.

        Args:
            init_mode: Initialization mode, "etcd" or "metastore"
            worker_port: DS worker port (required)
            worker_args: Additional worker arguments
            etcd_address: Etcd address (required for etcd mode)
            metastore_port: Metastore service port (required for metastore mode)

        Returns:
            Dict containing init_mode, worker_args, etcd/metastore_address, and node_worker_addresses
        """
        if self._initialized:
            # Log warning if parameters differ from initialized config
            if init_mode != self._init_mode or worker_args != self._worker_args:
                address_info = (
                    f"etcd={self._etcd_address}"
                    if self._init_mode == "etcd"
                    else f"metastore={self._metastore_address}"
                )
                logger.warning(
                    f"YR backend already initialized with mode={self._init_mode}, "
                    f"{address_info}, worker_args='{self._worker_args}'. "
                    f"Ignoring new parameters: mode={init_mode}, worker_args='{worker_args}'"
                )
            return self._get_backend_info_dict()

        self._init_mode = init_mode
        self._worker_args = worker_args

        if init_mode == "etcd":
            if not etcd_address:
                raise RuntimeError("etcd mode requires etcd_address")
            self._etcd_address = etcd_address
            # worker_port can be 0 for auto-assignment in etcd mode
            return self._initialize_etcd_mode(worker_port, worker_args)

        elif init_mode == "metastore":
            if not metastore_port:
                raise RuntimeError("metastore mode requires metastore_port")
            return self._initialize_metastore_mode(
                worker_port, metastore_port, worker_args
            )

        else:
            raise RuntimeError(f"Unknown init_mode: {init_mode}")

    def _initialize_etcd_mode(self, worker_port: int, worker_args: str) -> dict:
        """Initialize YR backend using user-provided etcd.

        In etcd mode, all nodes start DS workers that connect to the same etcd.
        No head/worker distinction - all actors are equivalent.

        Args:
            worker_port: DS worker port
            worker_args: Additional worker arguments

        Returns:
            Dict containing backend info
        """
        alive_nodes = self._get_alive_nodes()
        node_ips = [n["NodeManagerAddress"] for n in alive_nodes]
        logger.info(f"Found {len(alive_nodes)} alive Ray nodes: {node_ips}")

        # Create placement group
        pg = self._create_placement_group(alive_nodes)

        # Create DataSystemActor on each node
        worker_actors = []
        for rank in range(len(alive_nodes)):
            node_ip = self._get_bundle_node_ip(pg, rank)
            actor_name = f"DataSystemActor_{node_ip}_{worker_port}"
            actor = DataSystemActor.options(  # type: ignore[attr-defined]
                scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=rank,
                    placement_group_capture_child_tasks=False,
                ),
                lifetime="detached",
                name=actor_name,
                namespace="yr_backend",
            ).remote(
                init_mode="etcd",
                etcd_address=self._etcd_address,
                worker_port=worker_port,
                worker_args=worker_args,
            )
            worker_actors.append(actor)

        logger.info(f"Created {len(worker_actors)} DataSystemActor instances")

        # Start all workers in parallel
        logger.info("Starting all DS workers in parallel...")
        ray.get([actor.start.remote() for actor in worker_actors])

        # Collect node worker addresses
        self._collect_worker_addresses(worker_actors)

        self._placement_group = pg
        self._worker_actors = worker_actors
        self._initialized = True

        logger.info(
            f"YR etcd backend started: etcd={self._etcd_address}, "
            f"workers on {len(node_ips)} nodes: {self._node_worker_addresses}"
        )

        return self._get_backend_info_dict()

    def _initialize_metastore_mode(
        self, worker_port: int, metastore_port: int, worker_args: str
    ) -> dict:
        """Initialize YR backend using metastore mode.

        In metastore mode, the head node starts a metastore service,
        and worker nodes connect to it. Must start head first to initialize
        metastore service before worker nodes can connect.

        Args:
            worker_port: DS worker port
            metastore_port: Metastore service port
            worker_args: Additional worker arguments

        Returns:
            Dict containing backend info
        """
        alive_nodes = self._get_alive_nodes()
        node_ips = [n["NodeManagerAddress"] for n in alive_nodes]
        logger.info(f"Found {len(alive_nodes)} alive Ray nodes: {node_ips}")

        # Create placement group
        pg = self._create_placement_group(alive_nodes)

        # Dynamically determine head node from bundle 0
        head_ip = self._get_bundle_node_ip(pg, 0)
        metastore_address = f"{head_ip}:{metastore_port}"
        logger.info(f"Head node dynamically assigned to bundle 0: {head_ip}")

        # Create head actor (bundle 0)
        head_actor_name = f"DataSystemActor_{head_ip}_{worker_port}_head"
        head_actor = DataSystemActor.options(  # type: ignore[attr-defined]
            scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=0,
                placement_group_capture_child_tasks=False,
            ),
            lifetime="detached",
            name=head_actor_name,
            namespace="yr_backend",
        ).remote(
            init_mode="metastore",
            metastore_address=metastore_address,
            is_head=True,
            worker_args=worker_args,
            worker_port=worker_port,
        )

        # Start head actor first to initialize metastore service
        logger.info("Starting head worker to initialize metastore...")
        ray.get(head_actor.start.remote())
        self._metastore_address = metastore_address
        logger.info(
            f"Head worker started, metastore address: {self._metastore_address}"
        )

        # Create and start worker actors (bundle 1+)
        all_actors = [head_actor]
        worker_actors = []
        for idx in range(1, len(alive_nodes)):
            node_ip = self._get_bundle_node_ip(pg, idx)
            actor_name = f"DataSystemActor_{node_ip}_{worker_port}"

            actor = DataSystemActor.options(  # type: ignore[attr-defined]
                scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=idx,
                    placement_group_capture_child_tasks=False,
                ),
                lifetime="detached",
                name=actor_name,
                namespace="yr_backend",
            ).remote(
                init_mode="metastore",
                metastore_address=metastore_address,
                is_head=False,
                worker_args=worker_args,
                worker_port=worker_port,
            )
            worker_actors.append(actor)
            all_actors.append(actor)

        # Start worker actors in parallel (after head is ready)
        if worker_actors:
            logger.info(f"Starting {len(worker_actors)} worker actors in parallel...")
            ray.get([actor.start.remote() for actor in worker_actors])

        # Collect node worker addresses
        self._collect_worker_addresses(all_actors)

        self._placement_group = pg
        self._worker_actors = all_actors
        self._initialized = True

        logger.info(
            f"YR metastore backend started: metastore at {self._metastore_address}, "
            f"workers on {len(node_ips)} nodes"
        )

        return self._get_backend_info_dict()

    def get_backend_info(
        self,
        init_mode: Optional[str] = None,
        worker_port: Optional[int] = None,
        worker_args: Optional[str] = None,
        etcd_address: Optional[str] = None,
        metastore_port: Optional[int] = None,
    ) -> dict:
        """Get backend info.

        Behavior depends on initialization state and parameters:
        - If not initialized and parameters provided: initialize first
        - If not initialized and no parameters: raise RuntimeError
        - If already initialized: return info (parameters ignored)

        Multiple drivers calling this method concurrently will share the same
        backend. First call initializes, subsequent calls return existing config.

        Args:
            init_mode: Initialization mode (required if not initialized)
            worker_port: DS worker port (required if not initialized)
            worker_args: Additional worker arguments
            etcd_address: Etcd address (for etcd mode)
            metastore_port: Metastore port (for metastore mode)

        Returns:
            Dict containing init_mode, worker_args, etcd/metastore_address, and node_worker_addresses

        Raises:
            RuntimeError: If not initialized and no parameters provided
        """
        if not self._initialized:
            if init_mode is None or worker_port is None:
                raise RuntimeError(
                    "YR backend not initialized. "
                    "Please call register_yr_tensor_transport() first."
                )
            return self.initialize(
                init_mode, worker_port, worker_args or "", etcd_address, metastore_port
            )
        return self._get_backend_info_dict()


def ensure_yr_backend_initialized(
    init_mode: Optional[str] = None,
    worker_port: Optional[int] = None,
    worker_args: Optional[str] = None,
    etcd_address: Optional[str] = None,
    metastore_port: Optional[int] = None,
) -> dict:
    """Ensure YR backend is initialized.

    Returns:
        backend_info dict containing init_mode, worker_args, etcd/metastore_address,
        and node_worker_addresses
    """
    init_mode_val = init_mode or os.getenv("YR_DS_INIT_MODE") or "metastore"
    worker_port_val = worker_port or int(os.getenv("YR_DS_WORKER_PORT", "31501"))

    if init_mode_val == "metastore":
        metastore_port_val = metastore_port or int(
            os.getenv("YR_DS_METASTORE_PORT", "2379")
        )
        etcd_address_val = None
    elif init_mode_val == "etcd":
        metastore_port_val = None
        etcd_address_val = etcd_address or os.getenv("YR_DS_ETCD_ADDRESS")
    else:
        raise RuntimeError(
            f"Unknown init_mode: {init_mode_val}. Must be 'etcd' or 'metastore'."
        )

    worker_args_val = worker_args or os.getenv("YR_DS_WORKER_ARGS") or ""

    coordinator = YRBackendCoordinator.options(  # type: ignore[attr-defined]
        namespace="yr_backend", get_if_exists=True
    ).remote()

    backend_info: dict = ray.get(
        coordinator.get_backend_info.remote(
            init_mode_val,
            worker_port_val,
            worker_args_val,
            etcd_address_val,
            metastore_port_val,
        )
    )
    return backend_info


def get_yr_backend_info() -> dict:
    """Get YR backend info if already initialized.

    This function is used internally by YRTensorTransport to get worker addresses.
    Does NOT trigger initialization.

    Returns:
        backend_info dict containing init_mode, worker_args, etcd/metastore_address,
        and node_worker_addresses

    Raises:
        RuntimeError: If YR backend is not initialized
    """
    coordinator = YRBackendCoordinator.options(  # type: ignore[attr-defined]
        namespace="yr_backend", get_if_exists=True
    ).remote()

    backend_info: dict = ray.get(coordinator.get_backend_info.remote())
    return backend_info
