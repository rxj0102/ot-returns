"""Placeholder tests for otreturns.factors — implemented in Prompt 7."""

import pytest
import numpy as np
from otreturns.factors import decompose_transport_map


def test_decompose_not_implemented():
    with pytest.raises(NotImplementedError):
        decompose_transport_map(
            np.linspace(-1, 1, 100),
            np.linspace(-0.5, 1.5, 100),
            np.random.default_rng(0).normal(0, 1, (500, 3)),
        )
