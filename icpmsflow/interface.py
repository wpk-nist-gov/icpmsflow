"""
simplified interface for analysis
"""


import numpy as np
import pandas as pd

from .cached_decorators import gcached
from .core import (
    DataApplier,
    _get_col_or_level,
    cumtrapz_frame,
    load_paths,
    norm_frame,
    tidy_frame,
)
from .plotbounds import DataExplorerCombined


class ICPMSAnalysis(DataApplier):

    _NEW_LIKE_KEYS = [
        "x_dim",
        "batch_dim",
        "element_dim",
        "value_dim",
        "is_tidy",
        "drop_unused",
        "bounds_data",
        "meta",
    ]

    def __init__(
        self,
        frame,
        x_dim="Time [Sec]",
        batch_dim="batch",
        value_dim="value",
        element_dim="element",
        is_tidy=False,
        drop_unused=False,
        bounds_data=None,
        meta=None,
        **kws,
    ):

        self.frame = frame
        self.x_dim = x_dim
        self.batch_dim = batch_dim
        self.value_dim = value_dim
        self.element_dim = element_dim

        self.is_tidy = is_tidy
        self.drop_unused = drop_unused
        self.meta = meta
        self.bounds_data = bounds_data
        self.kws = kws

    # mimic DataApplier interface
    @property
    def y_dim(self):
        if self.is_tidy:
            return self.value_dim
        else:
            return None

    @property
    def by(self):
        if self.is_tidy:
            return [self.batch_dim, self.element_dim]
        else:
            return [self.batch_dim]

    @property
    def var_dim(self):
        """default dimension name var variables"""
        return self.element_dim

    @property
    def batches(self):
        return _get_col_or_level(self.frame, self.batch_dim).unique()

    @property
    def elements(self):
        if self.is_tidy:
            out = _get_col_or_level(self.frame, self.element_dim).unique()
        else:
            exclude = [self.x_dim] + self.by
            out = np.unique([x for x in self.frame.columns if x not in exclude])
        return out

    # Have to redo tidy here.
    def tidy(self, value_dim=None, var_dim=None, id_dims=None, **kws):

        if value_dim is None:
            value_dim = self.value_dim

        if var_dim is None:
            var_dim = self.var_dim

        by = [] if self.by is None else self.by
        x_dim = [] if self.x_dim is None else [self.x_dim]

        if id_dims is None:
            id_dims = by + x_dim

        data_tidy = tidy_frame(
            self.frame, value_name=value_dim, var_name=var_dim, id_vars=id_dims, **kws
        )
        return self.new_like(
            frame=data_tidy,
            x_dim=self.x_dim,
            value_dim=value_dim,
            is_tidy=True,
        )

    def add_bounds(
        self,
        kernel_size=None,
        y_dim=None,
        mean_over=None,
        tidy_kws=None,
        type_name="type_bound",
        lower_name="lower_bound",
        upper_name="upper_bound",
        baseline_name="baseline",
        signal_name="signal",
        z_threshold=None,
    ):

        bounds_data = self.get_bounds(
            kernel_size=kernel_size,
            y_dim=y_dim,
            mean_over=mean_over,
            tidy_kws=tidy_kws,
            type_name=type_name,
            lower_name=lower_name,
            upper_name=upper_name,
            baseline_name=baseline_name,
            signal_name=signal_name,
            z_threshold=z_threshold,
        )

        return self.new_like(bounds_data=bounds_data)

    # other utilies
    def get_bounds(
        self,
        kernel_size=None,
        y_dim=None,
        mean_over=None,
        tidy_kws=None,
        type_name="type_bound",
        lower_name="lower_bound",
        upper_name="upper_bound",
        baseline_name="baseline",
        signal_name="signal",
        z_threshold=None,
    ):

        data = self
        if kernel_size is not None:
            data = data.median_filter(kernel_size=kernel_size)
        data = data.gradient()

        lb = data.argmax()
        ub = data.argmin()

        if not self.is_tidy:
            if tidy_kws is None:
                tidy_kws = {}
            if y_dim is None:
                y_dim = tidy_kws.get("value_dim", self.value_dim)
            if mean_over is None:
                mean_over = tidy_kws.get("var_dim", self.var_dim)

            lb = lb.tidy(**tidy_kws)
            ub = ub.tidy(**tidy_kws)

        else:
            y_dim = self.y_dim
            if mean_over is None:
                mean_over = self.var_dim

        # mean over stuff
        by = [k for k in self.by if k != mean_over]
        lb = lb.frame.drop(mean_over, axis=1).groupby(by)  # .mean()
        ub = ub.frame.drop(mean_over, axis=1).groupby(by)  # .mean()

        if z_threshold is None:
            lb = lb.mean()
            ub = ub.mean()

        else:
            # apply z score
            from scipy.stats import zscore

            func = lambda x: x.where(np.abs(zscore(x.values)) <= z_threshold).mean()
            lb = lb.apply(func)
            ub = ub.apply(func)

        # make a frame with lower and upper bounds combined
        df = pd.merge(
            lb.rename(columns={y_dim: "lb"}),
            ub.rename(columns={y_dim: "ub"}),
            left_index=True,
            right_index=True,
        )

        df = df.fillna(0.0)

        baseline = (
            df.assign(type_bound=baseline_name)
            .assign(lower_bound=0.0)
            .assign(upper_bound=lambda x_dim: x_dim["lb"])[
                [type_name, lower_name, upper_name]
            ]
            .set_index("type_bound", append=True)
        )

        signal = (
            df.assign(type_bound=signal_name)
            .assign(lower_bound=lambda x_dim: x_dim["lb"])
            .assign(upper_bound=lambda x_dim: x_dim["ub"])[
                [type_name, lower_name, upper_name]
            ]
            .set_index("type_bound", append=True)
        )
        return pd.concat((baseline, signal)).sort_index()

    # @gcached()
    @property
    def explorer(self):
        if not hasattr(self, "_explorer"):
            kws = {
                "data": self.tidy().frame,
                "time_dim": self.x_dim,
                "batch_dim": self.batch_dim,
                "element_dim": self.element_dim,
                "value_dim": self.value_dim,
            }

            if self.bounds_data is not None:
                kws["bounds_data"] = self.bounds_data
            self._explorer = DataExplorerCombined(**kws)
        return self._explorer

    def from_explorer(self, explorer=None):
        if explorer is None:
            explorer = self.explorer
        return self.new_like(bounds_data=explorer.bounds_data)

    def to_csv(self, basename, overwrite=False):
        """
        create csv files for frame, bounds_data, and meta


        Parameters
        ----------
        basename : str
        basename of output files.  Will create {basename}.csv, {basename}.bounds.csv, and {basename}.meta.csv

        """

        from pathlib import Path

        base = Path(basename)

        if base.suffix == ".csv":
            base = base.with_suffix("")

        mapping = [
            ("frame", ".csv", self.frame),
            ("bounds_data", ".bounds.csv", self.bounds_data),
            ("meta", ".meta.csv", self.meta),
        ]

        d = {}
        for name, suffix, frame in mapping:
            if frame is not None:
                path = base.with_suffix(suffix)
                if not overwrite and path.exists():
                    raise ValueError(f"{path} exists")

                d[name] = (path, frame)

        for name, (path, frame) in d.items():
            if name == "bounds_data":
                index = True
            else:
                index = False
            frame.to_csv(path, index=index)

    @classmethod
    def from_csv(cls, basename, **kws):
        from pathlib import Path

        base = Path(basename)

        if base.suffix == ".csv":
            base = base.with_suffix("")

        mapping = [
            ("frame", ".csv"),
            ("bounds_data", ".bounds.csv"),
            ("meta", ".meta.csv"),
        ]

        d = {}
        for name, suffix in mapping:
            path = base.with_suffix(suffix)
            if path.exists():
                d[name] = path

        if "frame" not in d:
            raise ValueError("no base frame found")

        for name, path in d.items():
            if name == "bounds_data":
                index_col = [0, 1]
            else:
                index_col = None
            kws[name] = pd.read_csv(path, index_col=index_col)

        return cls(**kws)

    # @property
    # def bounds_data(self):
    #     return self.data_explorer.bounds_data

    @classmethod
    def from_paths(cls, paths, load_kws=None, **kws):
        if load_kws is None:
            load_kws = {}

        df, df_meta = load_paths(paths, **load_kws)
        return cls(frame=df, meta=df_meta, **kws)


class ICPMSAnalysis_old:
    def __init__(
        self,
        data=None,
        batch_dim="batch",
        time_dim="Time [Sec]",
        element_dim="element",
        value_dim="intensity",
        valuetot_dim="value",
        vartot_dim="variable",
        meta=None,
        bounds_data=None,
    ):

        self.data = data
        self.batch_dim = batch_dim
        self.time_dim = time_dim
        self.element_dim = element_dim
        self.value_dim = value_dim
        self.valuetot_dim = valuetot_dim
        self.vartot_dim = vartot_dim
        self.meta = meta
        self._bounds_data = bounds_data

    @gcached()
    def data_cum(self):
        return cumtrapz_frame(self.data, x=self.time_dim, groupby=self.batch_dim)

    def _norm(self, data):
        return norm_frame(data, x=self.time_dim, groupby=self.batch_dim)

    @gcached()
    def data_norm(self):
        return self._norm(self.data)

    @gcached()
    def data_cum_norm(self):
        return self._norm(self.data_cum)

    def _tidy(self, data, value_name=None):
        return tidy_frame(
            data,
            value_name=value_name,
            id_vars=[self.batch_dim, self.time_dim],
            var_name=self.element_dim,
        )

    @gcached()
    def data_tidy(self):
        return self._tidy(self.data, value_name=self.value_dim)

    @gcached()
    def data_cum_tidy(self):
        return self._tidy(self.data_cum, self.value_dim + "_cum")

    @gcached()
    def data_norm_tidy(self):
        return self._tidy(self.data_norm, self.value_dim + "_norm")

    @gcached()
    def data_cum_norm_tidy(self):
        return self._tidy(self.data_cum_norm, self.value_dim + "_cum_norm")

    @gcached()
    def data_tot(self):
        return (
            self.data_tidy.merge(self.data_norm_tidy)
            .merge(self.data_cum_tidy)
            .merge(self.data_cum_norm_tidy)
        )

    @gcached()
    def data_tot_tidy(self):
        return tidy_frame(
            self.data_tot,
            id_vars=[self.batch_dim, self.element_dim, self.time_dim],
            var_name=self.vartot_dim,
            value_name=self.valuetot_dim,
        )

    @gcached()
    def data_explorer(self):
        kws = {
            "data": self.data_tot_tidy,
            "time_dim": self.time_dim,
            "batch_dim": self.batch_dim,
            "variable_dim": self.vartot_dim,
            "value_dim": self.valuetot_dim,
        }

        if self._bounds_data is not None:
            kws["bounds_data"] = self._bounds_data
        return DataExplorerCombined(**kws)

    @property
    def bounds_data(self):
        return self.data_explorer.bounds_data

    @classmethod
    def from_paths(cls, paths, load_kws=None, **kws):
        if load_kws is None:
            load_kws = {}

        df, df_meta = load_paths(paths, **load_kws)
        return cls(data=df, meta=df_meta, **kws)
