"""

Core routines to analyze icpms data

"""

from datetime import datetime
from pathlib import Path

import numpy as np

# import numpy as np
import pandas as pd

# from pathlib import Path
from scipy.integrate import cumtrapz, trapz
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


def read_csv_with_meta(path, ntop=3, nbot=1, sep=",", urls=False, **kws):
    """
    Rease csv file, metadata at top of file

    Parameters
    ----------
    ntop : int, default=3
        number of rows at top of file corresponding to metadata
    nbot : int, default=1
        number of rows (not counting blank lines) that correspond to metadata
    sep : string, default=','
    kws : dict
    urls : bool, default=False
        If True, then each path is treated as a webpage.  The data is directly read from the web

    Returns
    -------
    df : pandas.DataFrame
        DataFrame read in
    meta : list
        meta data from top and bottom rows



    """

    if urls:
        from urllib.request import urlopen as myopen
    else:
        myopen = lambda x: open(x, "r")

    with myopen(path) as f:
        # grap metadata from first rows
        meta = [f.readline().strip() for _ in range(ntop)]
        # clean up meta data
        meta = [x.decode() if isinstance(x, bytes) else x for x in meta]

        df = pd.read_csv(f, **kws)
    # grab last line

    for i, g in df.iloc[-nbot:, :].iterrows():
        s = sep.join(g.dropna().str.strip().values)
        meta.append(s)

    df = df.iloc[:-nbot].applymap(pd.to_numeric)

    return df, meta


def parse_meta_list(meta_list):
    """
    Parse list of (standard) meta_list data to dictonary

    Parameters
    ----------
    meta_list : list
        list of meta data. Expected form is:

        * meta_list[0] : data path
        * meta_list[1] : info
        * meta_list[2] : acquired time, e.g., "Acquired      : 2021-02-01 16:08:12 using Batch 1FEB21SC2402."
        * meta_list[3] : printed time, e.g., ""Printed:2021-02-01 16:10:51"

    Returns
    -------
    meta_dict : dict
        dictionary of meta data values

    """
    out = {}
    out["data_path"] = meta_list[0]
    out["info"] = meta_list[1].strip()
    time_batch = ":".join((meta_list[2].split(":"))[1:]).split("using Batch")
    out["aquired_time"] = time_batch[0].strip()
    out["batch"] = time_batch[1].strip()
    out["printed_time"] = ":".join(meta_list[3].split(":")[1:])
    # convert datetimes
    for k in ["aquired_time", "printed_time"]:
        out[k] = datetime.strptime(out[k], "%Y-%m-%d %H:%M:%S")
    return out


def move_columns_to_front(df, cols):
    if isinstance(cols, str):
        cols = [cols]
    return df[cols + list(df.columns.drop(cols))]


def load_paths(paths, index_cols=None, **kws):
    """
    load multiple csv files into single dataframe

    Parameters
    ----------
    paths : str, Path, or sequence of str or Path.
        collection of paths to read.  If str or Path, read in single
        path.  Otherwise read in multple paths
    meta_keys : string, or sequence of strings, optional
        keys to add to dataframe from meta dictionary
        default is to only include batch

    kws : dict
        extra arguemtns to `read_csv_with_meta`

    Returns
    -------
    df : pandas.DataFrame
        Dataframe of all read paths
    df_meta : pandas.DataFrame
        DataFrame of all metadata
    """

    if isinstance(paths, (str, Path)):
        paths = [paths]

    L = []
    M = []
    for path in paths:
        # analyze a single path
        df, meta_list = read_csv_with_meta(path, **kws)
        meta_dict = parse_meta_list(meta_list)
        meta_dict["read_path"] = str(path)
        L.append(df.assign(batch=meta_dict["batch"]))
        M.append(meta_dict)

    df = pd.concat(L, ignore_index=True)
    df_meta = pd.DataFrame(M)

    if index_cols is not None:
        df = df.set_index([x for x in index_cols if x in df.columns])
        df_meta = df_meta.set_index(
            [x for x in index_cols if x in df_meta.columns]
        )  # list(df_meta.columns.intersection(index_cols)))
    else:
        # make sure batch is first
        df = move_columns_to_front(df, "batch")
        df_meta = move_columns_to_front(df_meta, "batch")
    return df, df_meta


def tidy_frame(
    df,
    value_name="value",
    id_vars=["batch", "Time [Sec]"],
    value_vars=None,
    var_name="element",
    ignore_index=False,
    set_index=False,
    append=None,
    **kws,
):
    """
    pandas.DataFrame.melt with some convenient defaults

    Parameters
    ----------
    id_vars : tuple, list, or ndarray, optional
        Column(s) to use as identifier variables.
    value_vars : tuple, list, or ndarray, optional
        Column(s) to unpivot. If not specified, uses all columns that
        are not set as `id_vars`.
    var_name : scalar
        Name to use for the 'variable' column. If None it uses
        ``frame.columns.name`` or 'variable'.
    value_name : scalar, default 'value'
        Name to use for the 'value' column.
    ignore_index : bool, default True
        If True, original index is ignored. If False, the original index is retained.
        Index labels will be repeated as necessary.
    set_index : bool, default=False
        If True, add `id_vars` and `var_name` to index, if not already there
    append : bool, optional
        if `set_index`, this contols whether new levels are appended to the current
        index, or overwrite the current index.
        Default is to use the opposite of `ignore_index`

    kws : dict
        extra arguments to `df.melt`

    """
    if id_vars is not None:
        if isinstance(id_vars, str):
            id_vars = [id_vars]
        else:
            id_vars = list(id_vars)

        # if id_vars are already in index, then drop them
        id_vars = [x for x in id_vars if x not in df.index.names]
        if len(id_vars) == 0:
            id_vars = None

    out = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        value_name=value_name,
        var_name=var_name,
        ignore_index=ignore_index,
        **kws,
    )

    if set_index:
        if id_vars is None:
            id_vars = []
        cols = [k for k in id_vars + [var_name] if k not in df.index.names]

        if append is None:
            append = not ignore_index
        out = out.set_index(cols, append=append)

    return out


def _get_col_or_level(df, col):
    if col in df.columns:
        return df[col]
    else:
        return df.index.get_level_values(col)


def apply_func_over_groups(
    func,
    df,
    by=None,
    x_dim="Time [Sec]",
    y_dim=None,
    drop_unused=False,
    reduction=True,
    **kws,
):
    if y_dim is None:
        if by is None:
            col_drop = [x_dim]
        elif isinstance(by, str):
            col_drop = [x_dim, by]
        else:
            col_drop = [x_dim] + list(by)
        y_dim = [x for x in df.columns if x not in col_drop]
    elif isinstance(y_dim, str):
        y_dim = [y_dim]

    if by is None:
        if reduction:
            if drop_unused:
                out = df.iloc[[0], :].loc[:, y_dim].copy()
            else:
                out = df.iloc[[0], :].drop(x_dim, axis=1)
        else:
            if drop_unused:
                if x_dim in df.columns:
                    out = df.loc[:, [x_dim] + y_dim].copy()
                else:
                    out = df.loc[:, y_dim].copy()
            else:
                out = df.copy()

        out.loc[:, y_dim] = func(df, x_dim, y_dim, **kws)

        # xvals = _get_col_or_level(df, x_dim).values
        # out.loc[:, y_dim] = trapz(y_dim=df.loc[:, y_dim].values, x_dim=xvals, axis=0)
    else:
        out = pd.concat(
            (
                apply_func_over_groups(
                    func=func,
                    df=g,
                    by=None,
                    x_dim=x_dim,
                    y_dim=y_dim,
                    drop_unused=drop_unused,
                    reduction=reduction,
                    **kws,
                )
                for _, g in df.groupby(by, sort=False)
            )
        )
    return out


_APPLY_OVER_GROUPS_DEFAULT_DICT = {
    "Default_Parameters": """df : dataframe
    by : str or array of str, optional
        groupby keys.  If specified, apply {name} over each group. If not specified,
        apply {name} over entire frame
    x_dim : str
        column to use as "x_dim" values
    y_dim : str or array of str, optional
        columns of "y_dim" data.  If not specified,
        use all columns except `x_dim`
    drop_unused : bool
        If True, drop unused columns.  Otherwise, keep unused columns""",
    "Default_Returns": """output : pandas.DataFrame
        DataFrame with {name} applied over `y_dim`""",
    "Default_See_Also": "apply_func_over_groups",
}

_APPLY_OVER_GROUPS_DEFAULT_DOC = """{summary}

    Parameters
    ----------
    {Default_Parameters}

    Returns
    -------
    {Default_Returns}

    See Also
    --------
    {Default_See_Also}
"""


def doc_apply_over_groups(name="function", summary="apply {name} over `df`", **kws):
    def wrapped(func):
        doc = func.__doc__
        if doc is None:
            doc = _APPLY_OVER_GROUPS_DEFAULT_DOC
        func.__doc__ = doc.format(
            **_APPLY_OVER_GROUPS_DEFAULT_DICT, **kws, name=name, summary=summary
        ).format(name=name, summary=summary)
        return func

    return wrapped


def doc_apply_like(other_func):
    def wrapped(func):
        doc = (
            other_func.__doc__.replace("df : dataframe\n    ", "").replace(
                "frame", "self.frame"
            )
            + f"    {other_func.__name__}"
        )
        func.__doc__ = doc

        return func

    return wrapped


def _func_trapz(df, x, y):
    xvals = _get_col_or_level(df, x).values
    return trapz(y=df.loc[:, y].values, x=xvals, axis=0)


def _func_cumtrapz(df, x, y):
    xvals = _get_col_or_level(df, x).values
    return cumtrapz(y=df.loc[:, y].values, x=xvals, axis=0, initial=0)


def _func_norm(df, x, y):
    t = df.loc[:, y]
    return t / t.max()


def _func_median_filter(df, x, y, **kws):
    return median_filter(df.loc[:, y].values, **kws)


def _func_gradient(df, x, y, **kws):
    xvals = _get_col_or_level(df, x).values
    return np.gradient(df.loc[:, y].values, xvals, axis=0, **kws)


def _func_savgol_filter(df, x, y, **kws):
    return savgol_filter(df.loc[:, y].values, axis=0, **kws)


def _func_argmax(df, x, y, **kws):
    """
    find xvalue for maximum value
    """
    xvals = _get_col_or_level(df, x).values

    idx = np.argmax(df.loc[:, y].values, axis=0)

    return xvals[idx]


def _func_argmin(df, x, y, **kws):

    xvals = _get_col_or_level(df, x).values

    idx = np.argmin(df.loc[:, y].values, axis=0)

    return xvals[idx]


def _func_interpolate_na(df, x, y, **kws):
    """
    This assumes x is the index
    """

    if df.index.name != x:
        raise ValueError(f"{df.index.name} must by same as {x}")

    return df[y].interpolate(**kws)


@doc_apply_over_groups("argmin")
def argmin_frame(df, by=None, x_dim="Time [Sec]", y_dim=None, drop_unused=False):
    return apply_func_over_groups(
        func=_func_argmin,
        df=df,
        by=by,
        x_dim=x_dim,
        y_dim=y_dim,
        drop_unused=drop_unused,
        reduction=True,
    )


@doc_apply_over_groups("argmax")
def argmax_frame(df, by=None, x_dim="Time [Sec]", y_dim=None, drop_unused=False):
    return apply_func_over_groups(
        func=_func_argmax,
        df=df,
        by=by,
        x_dim=x_dim,
        y_dim=y_dim,
        drop_unused=drop_unused,
        reduction=True,
    )


@doc_apply_over_groups("interpolate_na")
def interpolate_na_frame(
    df,
    by=None,
    x_dim="Time [Sec]",
    y_dim=None,
    drop_unused=False,
    sort_index=False,
    restore_index=False,
    **kws,
):
    """replace nan values by interpolating y(x)

    Parameters
    ----------
    {Default_Parameters}
    sort_index : bool, default=False
        if True, then sort index before before interpolation
    restore_index : bool, default=False
        if True, restore index to be like `df`

    Returns
    -------
    {Default_Returns}

    See Also
    --------
    pandas.DataFrame.interpolate
    """

    # see if have multiindex
    df_in = df
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    df = df.set_index(x_dim)

    if sort_index:
        df = df.sort_index()

    kws = dict(dict(method="index"), **kws)

    out = apply_func_over_groups(
        func=_func_interpolate_na,
        df=df,
        by=by,
        x_dim=x_dim,
        y_dim=y_dim,
        drop_unused=drop_unused,
        reduction=False,
        **kws,
    )

    if restore_index:
        if isinstance(df_in.index, pd.MultiIndex):
            out = out.reset_index().set_index(df_in.index.names)
        elif df_in.index.name is not None:
            out = out.reset_index().set_index(df_in.index.name)
    return out


@doc_apply_over_groups("trapz")
def trapz_frame(df, by=None, x_dim="Time [Sec]", y_dim=None, drop_unused=False):
    return apply_func_over_groups(
        func=_func_trapz,
        df=df,
        by=by,
        x_dim=x_dim,
        y_dim=y_dim,
        drop_unused=drop_unused,
        reduction=True,
    )


@doc_apply_over_groups("cumtrapz")
def cumtrapz_frame(df, by=None, x_dim="Time [Sec]", y_dim=None, drop_unused=False):
    return apply_func_over_groups(
        func=_func_cumtrapz,
        df=df,
        by=by,
        x_dim=x_dim,
        y_dim=y_dim,
        drop_unused=drop_unused,
        reduction=False,
    )


@doc_apply_over_groups("normalize", summary="normalize by max value")
def norm_frame(df, by=None, x_dim="Time [Sec]", y_dim=None, drop_unused=False):
    return apply_func_over_groups(
        func=_func_norm,
        df=df,
        by=by,
        x_dim=x_dim,
        y_dim=y_dim,
        drop_unused=drop_unused,
        reduction=False,
    )


@doc_apply_over_groups("gradient", summary="gradient of frame dy/dx")
def gradient_frame(
    df, by=None, x_dim="Time [Sec]", y_dim=None, drop_unused=False, **kws
):
    return apply_func_over_groups(
        func=_func_gradient,
        df=df,
        by=by,
        x_dim=x_dim,
        y_dim=y_dim,
        drop_unused=drop_unused,
        reduction=False,
    )


@doc_apply_over_groups("savgol filter")
def savegol_filter_frame(
    df,
    by=None,
    x_dim="Time [Sec]",
    y_dim=None,
    drop_unused=False,
    window_length=None,
    polyorder=2,
    **kws,
):
    """smooth data with savegol filter

    Parameters
    ----------
    {Default_Parameters}
    window_length : int, default=None
        window_length parameter to savgol filter
    polyorder : int, default=2
        polyorder parameter to savgol filter
    kws : dict
        extra arguments to savegol_filter

    Returns
    -------
    {Default_Returns}

    See Also
    --------
    {Default_See_Also}
    scipy.signal.savgol_filter
    """

    if window_length is None:
        return df

    kws = dict(window_length=window_length, polyorder=polyorder, **kws)

    return apply_func_over_groups(
        func=_func_savgol_filter,
        df=df,
        by=by,
        x_dim=x_dim,
        y_dim=y_dim,
        drop_unused=drop_unused,
        reduction=False,
        **kws,
    )


@doc_apply_over_groups("median filter")
def median_filter_frame(
    df,
    by=None,
    x_dim="Time [Sec]",
    y_dim=None,
    drop_unused=False,
    kernel_size=None,
    **kws,
):
    """smooth data with median filter

    Parameters
    ----------
    {Default_Parameters}
    kernel_size : int, default=None
        kernel size of median filter
    kws : dict
        extra arguments to median_filter

    Returns
    -------
    {Default_Returns}

    See Also
    --------
    {Default_See_Also}
    scipy.ndimage.median_filter
    """
    if kernel_size is None:
        return df
    else:
        kernel_size = int(kernel_size)

    kws = dict(dict(size=(kernel_size, 1), mode="constant"), **kws)

    return apply_func_over_groups(
        func=_func_median_filter,
        df=df,
        by=by,
        x_dim=x_dim,
        y_dim=y_dim,
        drop_unused=drop_unused,
        reduction=False,
        **kws,
    )


@doc_apply_over_groups("interpolate")
def interpolate_newx_frame(
    df,
    x,
    x_dim,
    by=None,
    by_kws=None,
    method="index",
    mask_minmax=True,
    sort_index=True,
    ret_all=False,
    **kwargs,
):
    """
    interpolate a dataframe at x

    Parameters
    ----------
    {Default_Parameters}
    x : array
      values to interpolate at
    by_kws : dict
        extra arguments to df.groupby
    method : str
      method for interpolation (see pandas.DataFrame.interpolate
      options)
    ret_all: bool
      if True, return all.
      else only at interpolated points "x"
    mask_minmax : bool, default=True
        if True, only interpolate to values `x` within bounds
    kwargs : dict
        extra arguments to `pandas.DataFrame.interpolate`

    Returns
    -------
    {Default_Returns}

    See Also
    --------
    pandas.DataFrame.interpolate

    """

    if by:
        if by_kws is None:
            by_kws = {}

        if isinstance(by, str):
            by = [by]

        # exclude = by + [x_dim]
        exclude = by
        cols = [k for k in df.columns if k not in exclude]

        out = df.groupby(by, **by_kws)[cols].apply(
            lambda g: interpolate_newx_frame(
                g,
                x=x,
                x_dim=x_dim,
                by=None,
                method=method,
                mask_minmax=mask_minmax,
                sort_index=sort_index,
                ret_all=ret_all,
                **kwargs,
            )
        )

    else:
        # get min/max values along x_dim
        if mask_minmax:
            minx = df[x_dim].min()
            maxx = df[x_dim].max()
            xx = x[(minx <= x) & (x <= maxx)]
        else:
            xx = x

        # set index
        t = df.set_index(x_dim)

        # new index from union of indicies
        idx_df = t.index
        idx_x = pd.Index(xx, name=idx_df.name)
        idx_union = idx_df.union(idx_x).drop_duplicates()

        # reindex
        t = t.reindex(idx_union)
        # end new way

        if sort_index:
            t = t.sort_index()

        # interpolate over index (linear interpolation)
        t = t.interpolate(method=method, **kwargs)

        if not ret_all:
            t = t.reindex(idx_x)

        # t = t.reset_index()#x_dim)

        out = t

    return out


# Routines to handle DataApplier operations
class _CallableResult(object):
    def __init__(self, parent, series):
        self._parent = parent
        self._series = series

    def __call__(self, *args, **kwargs):
        return self._parent.new_like(self._series(*args, **kwargs))


class _Groupby(object):
    def __init__(self, parent, group):
        self._parent = parent
        self._group = group

    def __iter__(self):
        return ((meta, self._parent.new_like(x)) for meta, x in self._group)

    def __getattr__(self, attr):
        if hasattr(self._group, attr):
            out = getattr(self._group, attr)
            if callable(out):
                return _CallableResult(self._parent, out)
            else:
                return self._parent.new_like(out)
        else:
            raise AttributeError("no attribute {} in groupby".format(attr))


class _LocIndexer(object):
    def __init__(self, parent):
        self._parent = parent
        self._loc = self._parent.frame.loc

    def __getitem__(self, idx):
        out = self._loc[idx]
        if isinstance(out, (pd.Series, pd.DataFrame)):
            out = self._parent.new_like(out)
        return out

    def __setitem__(self, idx, values):
        self._parent.frame.loc[idx] = values


class _iLocIndexer(object):
    def __init__(self, parent):
        self._parent = parent
        self._iloc = self._parent.frame.iloc

    def __getitem__(self, idx):
        out = self._iloc[idx]
        if isinstance(out, (pd.Series, pd.DataFrame)):
            out = self._parent.new_like(out)
        return out

    def __setitem__(self, idx, values):
        self._parent.frame.iloc[idx] = values


class _Query(object):
    def __init__(self, parent):
        self._parent = parent
        self._frame = parent.frame

    def __call__(self, expr, **kwargs):
        return self._parent.new_like(self._frame.query(expr, **kwargs))


class DataApplier(object):

    _NEW_LIKE_KEYS = ["by", "x_dim", "y_dim", "drop_unused", "value_dim", "var_dim"]

    def __init__(
        self,
        frame,
        by=None,
        x_dim="Time [Sec]",
        y_dim=None,
        drop_unused=False,
        value_dim="intensity",
        var_dim="element",
        **kws,
    ):

        assert isinstance(frame, pd.DataFrame)

        self.frame = frame
        if isinstance(by, str):
            by = [by]
        self.by = by
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.drop_unused = drop_unused
        self.value_dim = value_dim
        self.var_dim = var_dim
        self.kws = kws

    def new_like(self, frame=None, **kws):
        if frame is None:
            frame = self.frame

        for k in self._NEW_LIKE_KEYS:
            if k not in kws:
                kws[k] = getattr(self, k)
        kws = dict(self.kws, **kws)
        return type(self)(frame, **kws)

    def _is_plain_index(self, index=None):
        if index is None:
            index = self.index
        return not isinstance(index, pd.MultiIndex) and index.name is None

    def set_index(self, keys=None, **kws):
        """
        set index of self.frame

        if keys is None, try to add self.by + [self.x_dim] to index
        """

        frame = self.frame
        # index = frame.index

        if keys is not None:
            out = frame.set_index(keys=keys, **kws)
        elif self._is_plain_index():
            out = frame.set_index(keys=self.by + [self.x_dim], **kws)
        else:
            out = frame.reset_index().set_index(self.by + [self.x_dim])
        return self.new_like(frame=out)

    def reset_index(self, **kws):
        return self.new_like(frame=self.frame.reset_index(**kws))

    def reindex_smart(self, index, union=False, sort=None, **kws):
        names = index.names

        if self._is_plain_index():
            frame = self.frame.set_index(names)
        else:
            frame = self.frame.reset_index().set_index(names)

        if union:
            index = frame.index.union(index, sort=sort)

        frame = frame.reindex(index, **kws)
        return self.new_like(frame=frame)

    # routines to apply transformations
    def _apply_func(
        self,
        _func,
        _reduction=False,
        as_frame=False,
        by=None,
        x_dim=None,
        y_dim=None,
        drop_unused=None,
        **kws,
    ):

        if by is None:
            by = self.by
        if x_dim is None:
            x_dim = self.x_dim
        if y_dim is None:
            y_dim = self.y_dim
        if drop_unused is None:
            drop_unused = self.drop_unused

        out = _func(
            df=self.frame,
            by=by,
            x_dim=x_dim,
            y_dim=y_dim,
            drop_unused=drop_unused,
            **kws,
        )

        if not as_frame:
            x_dim = None if _reduction else self.x_dim
            out = self.new_like(frame=out, x_dim=x_dim)

        return out

    @doc_apply_like(trapz_frame)
    def trapz(self, **kws):
        return self._apply_func(trapz_frame, True, **kws)

    @doc_apply_like(cumtrapz_frame)
    def cumtrapz(self, **kws):
        return self._apply_func(cumtrapz_frame, **kws)

    @doc_apply_like(cumtrapz_frame)
    def integrate(self, **kws):
        """alias for cumtrapz"""
        return self.cumtrapz(**kws)

    @doc_apply_like(norm_frame)
    def norm(self, **kws):
        return self._apply_func(norm_frame, **kws)

    @doc_apply_like(norm_frame)
    def normalize(self, **kws):
        return self.norm(**kws)

    @doc_apply_like(interpolate_na_frame)
    def interpolate_na(self, **kws):
        return self._apply_func(interpolate_na_frame, **kws)

    @doc_apply_like(gradient_frame)
    def gradient(self, **kws):
        return self._apply_func(gradient_frame, **kws)

    @doc_apply_like(savegol_filter_frame)
    def savgol_filter(self, window_length=None, polyorder=2, **kws):
        return self._apply_func(
            savegol_filter_frame,
            window_length=window_length,
            polyorder=polyorder,
            **kws,
        )

    @doc_apply_like(median_filter_frame)
    def median_filter(self, kernel_size=None, **kws):
        return self._apply_func(median_filter_frame, kernel_size=kernel_size, **kws)

    @doc_apply_like(argmin_frame)
    def argmin(self, **kws):
        return self._apply_func(argmin_frame, True, **kws)

    @doc_apply_like(argmax_frame)
    def argmax(self, **kws):
        return self._apply_func(argmax_frame, True, **kws)

    @doc_apply_like(tidy_frame)
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
            by=by + [var_dim],
            x_dim=self.x_dim,
            y_dim=value_dim,
        )

    # pandas wrapping
    def __repr__(self):
        return "<{}>\n".format(type(self).__name__) + repr(self.frame)

    # def _repr_html_(self):
    #     s = '<title>Applier</title>' + self.frame._repr_html_()
    #     return s

    # other utilies
    # def get_bounds(
    #     self,
    #     kernel_size=None,
    #     is_tidy=True,
    #     y_dim=None,
    #     mean_over=None,
    #     tidy_kws=None
    #     # z_threshold=None
    # ):

    #     data = self
    #     if kernel_size is not None:
    #         data = data.median_filter(kernel_size=kernel_size)
    #     data = data.gradient()

    #     lb = data.argmax()
    #     ub = data.argmin()

    #     if not is_tidy:
    #         if tidy_kws is None:
    #             tidy_kws = {}
    #         if y_dim is None:
    #             y_dim = tidy_kws.get("y_dim", "intensity")
    #         if mean_over is None:
    #             mean_over = tidy_kws.get("var_name", "element")

    #         lb = lb.tidy(**tidy_kws)
    #         ub = ub.tidy(**tidy_kws)

    #     else:
    #         if y_dim is None:
    #             y_dim = self.frame.columns.drop(self.by + [self.x_dim])[0]
    #         if mean_over is None:
    #             mean_over = self.by[-1]

    #     # mean over stuff
    #     by = [k for k in self.by if k != mean_over]
    #     lb = lb.frame.drop(mean_over, axis=1).groupby(by).mean()
    #     ub = ub.frame.drop(mean_over, axis=1).groupby(by).mean()

    #     # make a frame with lower and upper bounds combined
    #     df = pd.merge(
    #         lb.rename(columns={y_dim: "lb"}),
    #         ub.rename(columns={y_dim: "ub"}),
    #         left_index=True,
    #         right_index=True,
    #     )
    #     # return df

    #     baseline = (
    #         df.assign(type_bound="baseline")
    #         .assign(lower_bound=0.0)
    #         .assign(upper_bound=lambda x_dim: x_dim["lb"])[
    #             ["type_bound", "lower_bound", "upper_bound"]
    #         ]
    #     )

    #     signal = (
    #         df.assign(type_bound="signal")
    #         .assign(lower_bound=lambda x_dim: x_dim["lb"])
    #         .assign(upper_bound=lambda x_dim: x_dim["ub"])[
    #             ["type_bound", "lower_bound", "upper_bound"]
    #         ]
    #     )
    #     return pd.concat((baseline, signal)).sort_index()

    # pandas wrappers
    @property
    def plot(self):
        return self.frame.plot

    @property
    def hvplot(self):
        return self.frame.hvplot

    # def set_index(self, *args, **kwargs):
    #     return self.new_like(frame=self.frame.set_index(*args, **kwargs))

    def __iter__(self):
        return iter(self.frame)

    def __next__(self):
        return next(self.frame)

    @property
    def items(self):
        return self.frame.values

    @property
    def index(self):
        return self.frame.index

    # @property
    # def name(self):
    #     return self.frame.name

    # def copy(self):
    #     return self.__class__(data=self.s, base_class=self._base_class)

    def _wrapped_pandas_method(self, mtd, wrap=False, *args, **kwargs):
        """Wrap a generic pandas method to ensure it returns a GeoSeries"""
        val = getattr(self.frame, mtd)(*args, **kwargs)
        if wrap and isinstance(val, (pd.Series, pd.DataFrame)):
            val = self.new_like(val)
        return val

    def __getitem__(self, key):
        return self._wrapped_pandas_method("__getitem__", wrap=True, key=key)

    def xs(self, key, axis=0, level=None, drop_level=False, wrap=True):
        return self._wrapped_pandas_method(
            "xs", wrap=wrap, key=key, axis=axis, level=level, drop_level=drop_level
        )

    def __setitem__(self, idx, values):
        self.frame[idx] = values

    def append(
        self,
        to_append,
        ignore_index=False,
        verify_integrity=True,
    ):
        if isinstance(to_append, self.__class__):
            to_append = to_append.series

        s = self.frame.append(
            to_append, ignore_index=ignore_index, verify_integrity=verify_integrity
        )

        return self.new_like(s)

    def droplevel(self, level, axis=0):
        return self.new_like(self.frame.droplevel(level=level, axis=axis))

    def apply(self, func, convert_dtype=True, args=(), wrap=False, **kwds):

        return self._wrapped_pandas_method(
            "apply",
            wrap=wrap,
            func=func,
            convert_dtype=convert_dtype,
            args=args,
            **kwds,
        )

    def sort_index(self, wrap=True, *args, **kwargs):
        return self._wrapped_pandas_method("sort_index", wrap=wrap, *args, **kwargs)

    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=True,
        group_keys=True,
        # squeeze=False,
        observed=False,
        wrap=True,
        **kwargs,
    ):
        """
        wrapper around groupby.

        Paremters
        ---------
        wrap : bool, default=False
            if True, try to wrap output in class of self

        See Also
        --------
        `pandas.Series.groupby` documentation
        """

        group = self.s.groupby(
            by=by,
            axis=axis,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            # squeeze=squeeze,
            observed=observed,
            **kwargs,
        )
        if wrap:
            return _Groupby(self, group)
        else:
            return group

    @property
    def query(self):
        return _Query(self)

    @property
    def loc(self):
        return _LocIndexer(self)

    @property
    def iloc(self):
        return _iLocIndexer(self)

    # def groupby_allbut(self, drop, **kwargs):
    #     """
    #     groupby all but columns in drop
    #     """
    #     if not isinstance(drop, list):
    #         drop = [drop]
    #     by = _allbut(self.index.names, *drop)
    #     return self.groupby(by=by, **kwargs)

    # @classmethod
    # def _concat_to_series(cls, objs, **concat_kws):
    #     from collections.abc import Sequence, Mapping
    #     if isinstance(objs, Sequence):
    #         first = objs[0]
    #         if isinstance(first, cls):
    #             objs = (x._series for x in objs)
    #     elif isinstance(objs, Mapping):
    #         out = {}
    #         remap = None
    #         for k in objs:
    #             v = objs[k]
    #             if remap is None:
    #                 if isinstance(v, cls):
    #                     remap = True
    #                 else:
    #                     remap = False
    #             if remap:
    #                 out[k] = v._series
    #             else:
    #                 out[k] = v
    #         objs = out
    #     else:
    #         raise ValueError('bad input type {}'.format(type(first)))
    #     return pd.concat(objs, **concat_kws)

    # def concat_like(self, objs, **concat_kws):
    #     s = self._concat_to_series(objs, **concat_kws)
    #     return self.new_like(s)

    # @classmethod
    # def concat(cls, objs, concat_kws=None, *args, **kwargs):
    #     if concat_kws is None:
    #         concat_kws = {}
    #     s = cls._concat_to_series(objs, **concat_kws)
    #     return cls(s, *args, **kwargs)


def _allbut(levels, *names):
    names = set(names)
    return [item for item in levels if item not in names]


def parse_element(element, sep="->"):
    """parse an array of elements to element, number

    for example, parse_element('Li7 -> 7') returns ('Li7', 7)
    """
    tag, number = map(str.strip, element.split(sep))
    number = int(number)
    return tag, number


def parse_element_series(elements, sep="->", as_dict=False):
    s = elements.str.split(sep)
    tag = s.str.get(0).str.strip()
    number = s.str.get(1).astype(int)
    return tag, number


def trapz_frame2(df, groupby=None, x="Time [Sec]", y=None, drop_unused=False):
    """
    Perform integration on dataframe

    Parameter
    ---------
    df : dataframe
    grouby : str or array of str, optional
        names to groupby.  If not specified,
        integral over entire frame
    x : str
        column to use as "x" values
    y : str or array of str, optional
        columns of "y" data.  If not specified,
        use all columns except `x`

    Returns
    -------
    output : dataframe
        dataframe integrated over `x`
    """
    # total integrals
    if y is None:
        if groupby is None:
            col_drop = [x]
        elif isinstance(groupby, str):
            col_drop = [x, groupby]
        else:
            col_drop = [x] + list(groupby)
        y = [x for x in df.columns if x not in col_drop]
    elif isinstance(y, str):
        y = [y]

    if groupby is None:
        if drop_unused:
            out = df.iloc[[0], :].loc[:, y].copy()
        else:
            out = df.iloc[[0], :].drop(x, axis=1)

        xvals = _get_col_or_level(df, x).values

        out.loc[:, y] = trapz(y=df.loc[:, y].values, x=xvals, axis=0)
    else:
        out = pd.concat(
            (
                trapz_frame(g, groupby=None, x=x, y=y)
                for _, g in df.groupby(groupby, sort=False)
            )
        )

    return out


def cumtrapz_frame2(df, groupby=None, x="Time [Sec]", y=None, drop_unused=False):
    """
    Perform cumulative integration on dataframe

    Parameter
    ---------
    df : dataframe
    groupby : str or array of str, optional
        names to groupgroupby.  If not specified,
        integral over entire frame
    x : str
        column to use as "x" values
    y : str or array of str, optional
        columns of "y" data.  If not specified,
        use all columns except `x`

    Returns
    -------
    output : dataframe
        dataframe integrated over `x`
    """

    if y is None:
        if groupby is None:
            col_drop = [x]
        elif isinstance(groupby, str):
            col_drop = [x, groupby]
        else:
            col_drop = [x] + list(groupby)
        y = [x for x in df.columns if x not in col_drop]
    elif isinstance(y, str):
        y = [y]

    if groupby is None:
        if drop_unused:
            if x in df.columns:
                out = df.loc[:, [x] + y].copy()
            else:
                out = df.loc[:, y].copy()
        else:
            out = df.copy()

        xvals = _get_col_or_level(df, x).values

        out.loc[:, y] = cumtrapz(y=df.loc[:, y].values, x=xvals, initial=0, axis=0)

    else:
        out = pd.concat(
            (
                cumtrapz_frame(g, groupby=None, x=x, y=y, drop_unused=drop_unused)
                for _, g in df.groupby(groupby, sort=False)
            )
        )
    return out


def norm_frame2(df, groupby=None, x="Time [Sec]", y=None, drop_unused=False):
    """
    Perform normalization

    Parameter
    ---------
    df : dataframe
    groupby : str or array of str, optional
        names to groupgroupby.  If not specified,
        integral over entire frame
    x : str
        column to use as "x" values
        These are ignored
    y : str or array of str, optional
        columns of "y" data.  If not specified,
        use all columns except `x`

    Returns
    -------
    output : dataframe
        dataframe integrated over `x`
    """

    if y is None:
        if groupby is None:
            col_drop = [x]
        elif isinstance(groupby, str):
            col_drop = [x, groupby]
        else:
            col_drop = [x] + list(groupby)
        y = [x for x in df.columns if x not in col_drop]
    elif isinstance(y, str):
        y = [y]

    if groupby is None:
        if drop_unused:
            if x in df.columns:
                out = df.loc[:, [x] + y].copy()
            else:
                out = df.loc[:, y].copy()
        else:
            out = df.copy()

        t = df.loc[:, y]
        out.loc[:, y] = t / t.max()

    else:
        out = pd.concat(
            (
                norm_frame(g, groupby=None, x=x, y=y, drop_unused=drop_unused)
                for _, g in df.groupby(groupby, sort=False)
            )
        )
    return out


def median_filter_frame2(
    df, groupby=None, x="Time [Sec]", y=None, drop_unused=False, kernel_size=None, **kws
):
    """
    Perform smoothing with median filter

    Parameter
    ---------
    df : dataframe
    groupby : str or array of str, optional
        names to groupgroupby.  If not specified,
        integral over entire frame
    x : str
        column to use as "x" values
        These are ignored
    y : str or array of str, optional
        columns of "y" data.  If not specified,
        use all columns except `x`
    kernel_size : int

    kws : dict
        extra arguments to `scipy.ndimage.median_filter`
        Default values are

        * size: (kernel_size, 1)
        * mode: 'constant'

    Returns
    -------
    output : dataframe
        dataframe integrated over `x`
    """

    if kernel_size is None:
        return df

    kws = dict(dict(size=(kernel_size, 1), mode="constant"), **kws)

    if y is None:
        if groupby is None:
            col_drop = [x]
        elif isinstance(groupby, str):
            col_drop = [x, groupby]
        else:
            col_drop = [x] + list(groupby)
        y = [x for x in df.columns if x not in col_drop]
    elif isinstance(y, str):
        y = [y]

    if groupby is None:
        if drop_unused:
            if x in df.columns:
                out = df.loc[:, [x] + y].copy()
            else:
                out = df.loc[:, y].copy()
        else:
            out = df.copy()

        out.loc[:, y] = median_filter(out.loc[:, y].values, **kws)

    else:
        out = pd.concat(
            (
                median_filter_frame(
                    g,
                    groupby=None,
                    x=x,
                    y=y,
                    drop_unused=drop_unused,
                    kernel_size=kernel_size,
                    **kws,
                )
                for _, g in df.groupby(groupby, sort=False)
            )
        )
    return out
