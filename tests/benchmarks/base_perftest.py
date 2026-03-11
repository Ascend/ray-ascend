from abc import ABC, abstractmethod


class RayAscendBandwidthTester(ABC):
    @abstractmethod
    def run_bandwidth_test(self):
        raise NotImplementedError(
            "run_bandwidth_test must be implemented by subclasses."
        )

    @abstractmethod
    def close(self):
        raise NotImplementedError("close must be implemented by subclasses.")
