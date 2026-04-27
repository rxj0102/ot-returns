"""
Experiment: Two-sample Wasserstein tests — size and power analysis.

Usage:
    python experiments/run_two_sample_tests.py
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic import generate_two_sample_test_data
from otreturns.distributions import EmpiricalDistribution
# from otreturns.testing import wasserstein_two_sample_test  # Prompt 5


def main():
    print("Two-sample test power analysis")
    for shift in [0.0, 0.1, 0.2, 0.5]:
        s1, s2 = generate_two_sample_test_data(
            n1=500, n2=500, shift=shift, seed=42
        )
        mu = EmpiricalDistribution(s1)
        nu = EmpiricalDistribution(s2)
        print(f"  shift={shift:.1f}: mu={mu}, nu={nu}")

    print("Two-sample testing not yet implemented (Prompt 5).")


if __name__ == "__main__":
    main()
