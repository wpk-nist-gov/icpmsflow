import numpy as np
import pandas as pd

import icpmsflow


def test_trapz(fixture):

    x, y0, y1, df = fixture.x, fixture.y0, fixture.y1, fixture.df

    ds = icpmsflow.ICPMSAnalysis(frame=df, x_dim="time")

    from scipy.integrate import trapz

    np.testing.assert_allclose(
        np.array([trapz(y0, x), trapz(y1, x)]), ds.trapz().tidy().frame.value.values
    )


def test_gradient(fixture):

    x, y0, y1, df = fixture.x, fixture.y0, fixture.y1, fixture.df
    ds = icpmsflow.ICPMSAnalysis(frame=df, x_dim="time")

    for col, y in zip(["y0", "y1"], [y0, y1]):
        np.testing.assert_allclose(ds.gradient()[col].frame.values, np.gradient(y, x))


def test_bounds(fixture):

    x, y0, y1, df = fixture.x, fixture.y0, fixture.y1, fixture.df
    ds = icpmsflow.ICPMSAnalysis(frame=df, x_dim="time")
    # test bounds
    bounds = ds.add_bounds().bounds_data

    target = pd.DataFrame(
        {
            "batch": "batch_0",
            "type_bound": ["baseline", "signal"],
            "lower_bound": [0.0, 6.0],
            "upper_bound": [6.0, 13.0],
        }
    ).set_index(["batch", "type_bound"])

    pd.testing.assert_frame_equal(bounds, target)


def test_final_result(fixture):

    x, y0, y1, df = fixture.x, fixture.y0, fixture.y1, fixture.df
    ds = icpmsflow.ICPMSAnalysis(frame=df, x_dim="time")

    value = ds.add_bounds().normalize_by_baseline().interpolate_at_bounds(as_delta=True)
    target = pd.DataFrame(
        {
            "batch": "batch_0",
            "type_bound": ["baseline", "signal"],
            "time": [6.0, 7.0],
            "y0": [0.0, 6.5],
            "y1": [0.0, 6.5],
        }
    ).set_index(["batch", "type_bound"])

    pd.testing.assert_frame_equal(value, target)
