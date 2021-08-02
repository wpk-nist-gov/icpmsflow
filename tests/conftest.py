import numpy as np
import pandas as pd
import pytest


def get_synthetic_data(
    n0=None, n1=None, n2=None, ntot=None, baseline=0, signal=1, baseline_end=None
):
    if n0 is None:
        n0 = ntot // 3

    if n1 is None:
        n1 = n0

    if n2 is None:
        n2 = ntot - (n0 + n1)

    if baseline_end is None:
        baseline_end = baseline

    return np.array([baseline] * n0 + [signal] * n1 + [baseline_end] * n2)


class SyntheticTestData:
    def __init__(self, n):
        self.n = n
        x = np.linspace(0, n - 1, n)

        y0 = get_synthetic_data(ntot=n, signal=1.0)
        y1 = get_synthetic_data(ntot=n, baseline=0.5, signal=1.5, baseline_end=0)

        df = pd.DataFrame({"batch": "batch_0", "time": x, "y0": y0, "y1": y1})

        self.x = x
        self.y0 = y0
        self.y1 = y1
        self.df = df


@pytest.fixture(params=[(21,)], scope="module")
def fixture(request):
    return SyntheticTestData(*request.param)
