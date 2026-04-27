"""Placeholder tests for otreturns.regimes — implemented in Prompt 6."""

import pytest
from otreturns.regimes import detect_regimes, compute_distance_matrix


def test_compute_distance_matrix_not_implemented():
    with pytest.raises((NotImplementedError, TypeError)):
        compute_distance_matrix(None)


def test_detect_regimes_not_implemented():
    with pytest.raises((NotImplementedError, TypeError)):
        detect_regimes(None)
