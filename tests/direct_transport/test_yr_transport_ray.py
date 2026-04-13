# Tests for Ray integration features of YR Tensor Transport.
# This file focuses on Ray-specific behaviors like actor health checks and
# remote method calls, keeping them separate from core YR transport logic.
from unittest.mock import MagicMock, patch

import pytest

from ray_ascend.direct_transport.yr_tensor_transport import YRTensorTransport


class TestActorHealthCheck:
    """Test actor health check functionality for Ray integration."""

    @pytest.fixture
    def transport(self):
        """Create YRTensorTransport instance."""
        return YRTensorTransport()

    @pytest.fixture
    def mock_actor(self):
        """Create a mock actor with ray call chain."""
        mock_actor = MagicMock()
        mock_ray_call_chain = MagicMock()
        mock_actor.__ray_call__ = mock_ray_call_chain

        mock_ray_call_chain.options.return_value = mock_ray_call_chain
        mock_ray_call_chain.remote.return_value = "mock_object_ref"
        return mock_actor

    def test_actor_has_tensor_transport_success(self, transport, mock_actor):
        """
        Test that actor_has_tensor_transport returns True when health check succeeds.
        Verifies correct Ray concurrency group usage and remote call pattern.
        """
        with patch("ray.get", return_value=True) as mock_ray_get:
            result = transport.actor_has_tensor_transport(mock_actor)

            assert result is True
            mock_ray_get.assert_called_once_with("mock_object_ref")

            mock_actor.__ray_call__.options.assert_called_once_with(
                concurrency_group="_ray_system"
            )
            mock_actor.__ray_call__.remote.assert_called_once()

    def test_actor_has_tensor_transport_failure(self, transport, mock_actor):
        """Test that actor_has_tensor_transport returns False when health check fails."""
        with patch("ray.get", return_value=False) as mock_ray_get:
            result = transport.actor_has_tensor_transport(mock_actor)

            assert result is False
            mock_ray_get.assert_called_once()
