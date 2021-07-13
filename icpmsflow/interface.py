"""
simplified interface for analysis
"""


import numpy as np
import pandas as pd

# from .cached_decorators import gcached
from .core import (  # cumtrapz_frame,; norm_frame,
    DataApplier,
    _get_col_or_level,
    load_paths,
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
        """
        Parameters
        ----------
        frame : pandas.DataFrame
            input frame
        x_dim : str, default="Time [Sec]"
            name of column with "x" values
        batch_dim : str, default="batch"
            name of column containing batch values
        value_dim : str, default='value'
            name of 'value' column in tidy format
        element_dim : str, default='element'
            name of 'element' column in tidy format
        is_tidy : bool, default=False
            True value indicates data is in 'tidy' format
        drop_unused : bool, default=False
            Parameter passed to reduction/manipulation routines.
            If True, remove unused columns after operation
        bounds_data : pandas.DataFrame, optional
            DataFrame containing bounds information.
        meta : pandas.DataFrame, optional
            metadata
        kws : dict
            optional parameters to pass along to new objects
        """
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
        """'value' dimension"""
        if self.is_tidy:
            return self.value_dim
        else:
            return None

    @property
    def by(self):
        """groupby dimensions"""
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
        """data to 'tidy' format

        Parameters
        ----------
        value_dim : str, optional
            Name to use for the 'value' column.  Defaults to `self.value_dim`
        var_dim : str, optional
            Name to use for the 'variable' column.  Defaults to `self.element_dim`
        id_dims : str, optional
            Column(s) to use as identifier variables
            Defaults to `self.by + self.x_dim`.
        kws : dict
            extra arguments to `tidy_frame`

        See Also
        --------
        `tidy_frame`, `pandas.melt`
        """

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

    @staticmethod
    def _reset_or_copy(frame, reset_index, copy, deep):
        out = frame
        if reset_index and isinstance(out.index, pd.MultiIndex):
            out = out.reset_index()
        elif copy:
            out = out.copy(deep=deep)
        return out

    def get_frame(self, reset_index=True, copy=False, deep=True):
        """
        get current self.frame

        Parameters
        ----------
        reset_index : bool, default=True
            If True, and `self.frame` has a multiindex, then get a copy of data
            with reset_index
        copy : bool, default=False
            If True, return a copy of `self.frame`.  Note that this is ignored if
            the index is reset.
        deep : bool, default=True
            If true, and `copy`, return a deep copy
        """
        return self._reset_or_copy(
            self.frame, reset_index=reset_index, copy=copy, deep=deep
        )

    def get_bounds(self, reset_index=True, copy=False, deep=True):
        """
        Get the current bounds data

        This is a convenience function to work with `bounds_data` in multiple formats.
        The default form of `bounds_data` is a tidy DataFrame with a multiindex.

        Parameters
        ----------
        reset_index : bool, default=True
            If true, return copy of data, without multiindex
        copy : bool
            Return a copy of `self.bounds_data`.  This is ignored if
            the index is reset
        deep : bool, default=True
            If true, and `copy`, return a deep copy
        """
        return self._reset_or_copy(
            self.bounds_data, reset_index=reset_index, copy=copy, deep=deep
        )

    def set_bounds(self, bounds_data, set_index=None, type_name="type_bound"):
        """
        create new ICPMSAnalysis object with `bounds_data`

        Expect that Bounds Data will have multiindex with levels [`self.batch_dim`, `type_name`]
        and columns ['lower_bounds','upper_bound]

        Parameters
        ----------
        bounds_data : DataFrame
            bounds_data for new object
        set_index : bool, default=None
            if `set_index`, then it is assumed the input `bounds_data object
            Does not have a multiindex.  The index for `new.bounds_data` will be set to
            [`self.batch_dim`, `type_name`].
            If `set_index` is None, then try to guess if bounds_data has correct index.
        type_name : str, default="type_bound"
            name of column (or index level) with name of bounds type

        Returns
        -------
        new : ICPMSAnalysis object
        """

        if set_index is None:
            set_index = not isinstance(bounds_data, pd.MultiIndex)
        if set_index:
            bounds_data = bounds_data.set_index([self.batch_dim, type_name])

        return self.new_like(bounds_data=bounds_data)

    # other utilies
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
        as_frame=False,
        shift=0.0,
    ):
        """
        Add bounds_data programmatically.

        Finds the maximum/minimum in the gradient of the signal.  This done on a per batch level.
        The average location of the extrema across elements gives the position of the bounds.


        Parameters
        ----------
        kernel_size : int, optional
            if present, apply self.median_filter with this kernel size
        y_dim : str, optional
            name of value dimension.  Make guess otherwise
        mean_over : str, optional
            name of dimensaion to average over.  Uses var_dim otherwise
        tidy_kws : dict, optional
            key word arguments to `self.tidy`
        type_name : str
            column name of bounds type
        lower_name, upper_name : str
            name of lower/upper bound
        baseline_name, signal_name : str
            name of baseline/signal
        z_threshold : float, optional
            If present, perform zscore over dataset to remove outliers
            Value of about 3 is usually good
        shift : float, tuple,  optional
            shift guessed bounds up/down by this factor.
            * If tuple of length 2:
            shift baseline upper - shift[0], signal_lower + shift[0], signal_upper - shift[1]
            * If tuple of length 3:
            baseline_upper - shift[0], signal_lower - shift[1], signal_upper - shift[0]
            * If float:
            baseline_upper - shift, signal_lower + shift, signal_upper - shift
        as_frame : bool, default=False
            if True, return dataframe of bounds
            if False, return object like self with bounds_data set from this function

        Returns
        -------
        out : pd.DataFrame
            bounds frame

        """

        if isinstance(shift, float):
            shift = (shift,) * 3
        else:
            shift = tuple(shift)
            if len(shift) == 2:
                shift = (shift[0], shift[0], shift[1])
            elif len(shift) != 3:
                raise ValueError(f"shift of length {len(shift)} should be 2, or 3")

        assert all((x >= 0 for x in shift))

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
            .assign(upper_bound=lambda x_dim: x_dim["lb"] - shift[0])[
                [type_name, lower_name, upper_name]
            ]
            .set_index("type_bound", append=True)
        )

        signal = (
            df.assign(type_bound=signal_name)
            .assign(lower_bound=lambda x_dim: x_dim["lb"] + shift[1])
            .assign(upper_bound=lambda x_dim: x_dim["ub"] - shift[2])[
                [type_name, lower_name, upper_name]
            ]
            .set_index("type_bound", append=True)
        )

        bounds_data = pd.concat((baseline, signal)).sort_index()

        if as_frame:
            return bounds_data
        else:
            return self.new_like(bounds_data=bounds_data)

    def snap_bounds_minmax(
        self,
        bounds_data=None,
        as_frame=False,
        lower_name="lower_bound",
        upper_name="upper_bound",
        type_name="type_bound",
    ):
        """
        adjust bounds to fit within min/max of self.frame[self.x_dim]
        """

        bounds_data = self._bounds_check_index(bounds_data, type_name)
        # if bounds_data is None:
        #     bounds_data = self.bounds_data

        minmax = self.frame.groupby(self.by)[self.x_dim].agg(["min", "max"])

        bounds_data = (
            bounds_data.merge(minmax, left_index=True, right_index=True)
            # lower bound
            .assign(
                **{
                    lower_name: lambda x: x[lower_name].where(
                        x[lower_name] >= x["min"], other=x["min"], axis=0
                    )
                }
            )
            # upper bound
            .assign(
                **{
                    upper_name: lambda x: x[upper_name].where(
                        x[upper_name] <= x["max"], other=x["max"], axis=0
                    )
                }
            )
            # upper > lower
            .assign(
                **{
                    upper_name: lambda x: x[upper_name].where(
                        x[upper_name] >= x[lower_name], other=x[lower_name], axis=0
                    )
                }
            ).drop(["min", "max"], axis=1)
        )

        if as_frame:
            return bounds_data

        else:
            return self.new_like(bounds_data=bounds_data)

    def snap_bounds_nearest(
        self,
        bounds_data=None,
        as_frame=False,
        lower_name="lower_bound",
        upper_name="upper_bound",
        type_name="type_bound",
        baseline_name="baseline",
        signal_name="signal",
        baseline_lower="ffill",
        baseline_upper="ffill",
        signal_lower="bfill",
        signal_upper="ffill",
    ):
        """
        adjust bounds to fit correspond to nearest x value
        """

        bounds_data = self._bounds_check_index(bounds_data, type_name)
        # if bounds_data is None:
        #     bounds_data = self.bounds_data

        m = bounds_data.unstack(type_name).copy()

        method_dict = {
            (lower_name, baseline_name): baseline_lower,
            (lower_name, signal_name): signal_lower,
            (upper_name, baseline_name): baseline_upper,
            (upper_name, signal_name): signal_upper,
        }

        for meta, g in self.frame.groupby(self.by):

            idx = g.reset_index().set_index(self.x_dim).index

            for (edge, kind), method in method_dict.items():
                value = m.loc[meta, (edge, kind)]

                indexer = idx.get_indexer([value], method=method)
                indexer[indexer < 0] = 0
                new_value = idx[indexer[0]]
                m.loc[meta, (edge, kind)] = new_value

        out = m.stack(type_name)

        if as_frame:
            return out
        else:
            return self.new_like(bounds_data=out)

    def _bounds_check_index(self, bounds_data, type_name):
        if bounds_data is None:
            if self.bounds_data is None:
                raise ValueError("trying to default to unset `self.bounds_data`")
            bounds_data = self.bounds_data

        if isinstance(bounds_data.index, pd.MultiIndex):
            names = bounds_data.index.names
            assert type_name in names and self.batch_dim in names
        else:
            bounds_data = bounds_data.set_index([self.batch_dim, type_name])
        return bounds_data

    def _bounds_melt(self, bounds_data=None, bound_name="edge"):
        if bounds_data is None:
            bounds_data = self.bounds_data
        return bounds_data.melt(
            var_name=bound_name, value_name=self.x_dim, ignore_index=False
        )

    def _bounds_to_index(
        self, bounds_data=None, bounds_melt=None, drop_duplicates=True, sort=True
    ):

        if bounds_melt is None:
            bounds_melt = self._bounds_melt(bounds_data=bounds_data)
        out = bounds_melt.reset_index().set_index(self.by + [self.x_dim]).index
        if sort:
            out = out.sort_values()
        if drop_duplicates:
            out = out.drop_duplicates()

        return out

    def reindex_with_bounds(
        self, bounds_data=None, bounds_melt=None, union=True, **kws
    ):
        index = self._bounds_to_index(bounds_data=bounds_data, bounds_melt=bounds_melt)
        return self.reindex_smart(index, union=union, **kws)

    def interpolate_at_bounds(
        self,
        bounds_data=None,
        integrate=True,
        integrate_kws=None,
        type_name="type_bound",
        bound_name="edge",
        interp_kws=None,
        merge=True,
        upper_name="upper_bound",
        lower_name="lower_bound",
        as_delta=False,
        reset_index=False,
    ):
        """
        interpolate `self.frame` at values in `bounds_data`

        Parameters
        ----------
        bounds_data : DataFrame, optional
            Defaults to `self.bounds_data`
        integrate : bool, default=True
            perform integration before analysis
        integrate_kws : dict
            extra arguments to `self.integrate`
        type_name, bound_name : str
            names of type of bound (e.g., baseline, signal) and bound edge column.  The later
            is the column name of the stacked  bounds_data frame
        interp_kws : dict, optional
            optinal arguments to `self.interpolate_na`
        merge : bool, default=True
            if True, merge the interpolated results with `bounds_data` and
            return this DataFrame.
        as_delta : bool, default=False
            if True, take difference of `upper_name` - `lower_name`
            Therefore, to get the integral from 'lower' to 'upper' bounds, call
            this method with integral and `as_delta = True`
        reset_index, : bool, default=False
            If True, reset_index of final result

        Returns
        -------
        output : ICPMSAnalysis or pd.DataFrame
            if `merge` then output is a DataFrame.  Otherwise, output is
            an ICPMSAnalaysis object interpolated at `bounds_data`
        """

        bounds_data = self._bounds_check_index(bounds_data, type_name)
        bounds_melt = self._bounds_melt(bounds_data=bounds_data, bound_name=bound_name)
        bounds_index = self._bounds_to_index(bounds_melt=bounds_melt)

        if integrate:
            integrate_kws = {} if integrate_kws is None else integrate_kws
            new = self.integrate(**integrate_kws)
        else:
            new = self

        new = new.reindex_smart(bounds_index, union=True)

        if np.any(new.frame.isnull()):
            if interp_kws is None:
                interp_kws = {}
            new = new.interpolate_na(**interp_kws)

        if merge:
            new = pd.merge(
                bounds_melt.reset_index(),
                new.frame.reset_index(),
                on=self.by + [self.x_dim],
                how="left",
            )

            new = new.set_index(self.by + [type_name, bound_name])

            if as_delta:
                upper = new.xs(upper_name, level=bound_name)
                lower = new.xs(lower_name, level=bound_name)
                new = upper - lower

        if reset_index:
            new = new.reset_index()

        return new

    def normalize_by_baseline(
        self,
        delta_frame=None,
        bounds_data=None,
        integrate=True,
        integrate_kws=None,
        lower_name="lower_bound",
        upper_name="upper_bound",
        type_name="type_bound",
        baseline_name="baseline",
        signal_name="signal",
        bound_name="edge",
        interp_kws=None,
    ):

        if delta_frame is None:
            delta_frame = self.interpolate_at_bounds(
                bounds_data=bounds_data,
                integrate=integrate,
                integrate_kws=integrate_kws,
                type_name=type_name,
                bound_name=bound_name,
                interp_kws=interp_kws,
                merge=True,
                upper_name=upper_name,
                lower_name=lower_name,
                as_delta=True,
            )

        baseline = delta_frame.xs(baseline_name, level=type_name)

        dt = baseline[self.x_dim]
        dy = baseline.drop(self.x_dim, axis=1)

        baseline_mean = dy.apply(lambda x: x / dt)

        new_frame = self.set_index().frame - baseline_mean

        return self.new_like(frame=new_frame.reset_index())

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

    @classmethod
    def from_paths(cls, paths, load_kws=None, **kws):
        """
        Create ICPMSAnalysis object from csv files


        Parameters
        ----------
        """
        if load_kws is None:
            load_kws = {}

        df, df_meta = load_paths(paths, **load_kws)
        return cls(frame=df, meta=df_meta, **kws)


# class ICPMSAnalysis_old:
#     def __init__(
#         self,
#         data=None,
#         batch_dim="batch",
#         time_dim="Time [Sec]",
#         element_dim="element",
#         value_dim="intensity",
#         valuetot_dim="value",
#         vartot_dim="variable",
#         meta=None,
#         bounds_data=None,
#     ):

#         self.data = data
#         self.batch_dim = batch_dim
#         self.time_dim = time_dim
#         self.element_dim = element_dim
#         self.value_dim = value_dim
#         self.valuetot_dim = valuetot_dim
#         self.vartot_dim = vartot_dim
#         self.meta = meta
#         self._bounds_data = bounds_data

#     @gcached()
#     def data_cum(self):
#         return cumtrapz_frame(self.data, x=self.time_dim, groupby=self.batch_dim)

#     def _norm(self, data):
#         return norm_frame(data, x=self.time_dim, groupby=self.batch_dim)

#     @gcached()
#     def data_norm(self):
#         return self._norm(self.data)

#     @gcached()
#     def data_cum_norm(self):
#         return self._norm(self.data_cum)

#     def _tidy(self, data, value_name=None):
#         return tidy_frame(
#             data,
#             value_name=value_name,
#             id_vars=[self.batch_dim, self.time_dim],
#             var_name=self.element_dim,
#         )

#     @gcached()
#     def data_tidy(self):
#         return self._tidy(self.data, value_name=self.value_dim)

#     @gcached()
#     def data_cum_tidy(self):
#         return self._tidy(self.data_cum, self.value_dim + "_cum")

#     @gcached()
#     def data_norm_tidy(self):
#         return self._tidy(self.data_norm, self.value_dim + "_norm")

#     @gcached()
#     def data_cum_norm_tidy(self):
#         return self._tidy(self.data_cum_norm, self.value_dim + "_cum_norm")

#     @gcached()
#     def data_tot(self):
#         return (
#             self.data_tidy.merge(self.data_norm_tidy)
#             .merge(self.data_cum_tidy)
#             .merge(self.data_cum_norm_tidy)
#         )

#     @gcached()
#     def data_tot_tidy(self):
#         return tidy_frame(
#             self.data_tot,
#             id_vars=[self.batch_dim, self.element_dim, self.time_dim],
#             var_name=self.vartot_dim,
#             value_name=self.valuetot_dim,
#         )

#     @gcached()
#     def data_explorer(self):
#         kws = {
#             "data": self.data_tot_tidy,
#             "time_dim": self.time_dim,
#             "batch_dim": self.batch_dim,
#             "variable_dim": self.vartot_dim,
#             "value_dim": self.valuetot_dim,
#         }

#         if self._bounds_data is not None:
#             kws["bounds_data"] = self._bounds_data
#         return DataExplorerCombined(**kws)

#     @property
#     def bounds_data(self):
#         return self.data_explorer.bounds_data

#     @classmethod
#     def from_paths(cls, paths, load_kws=None, **kws):
#         if load_kws is None:
#             load_kws = {}

#         df, df_meta = load_paths(paths, **load_kws)
#         return cls(data=df, meta=df_meta, **kws)
