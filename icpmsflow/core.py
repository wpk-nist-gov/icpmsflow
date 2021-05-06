"""

Core routines to analyze icpms data

"""

from datetime import datetime

import numpy as np

# import numpy as np
import pandas as pd

# from pathlib import Path
from scipy.integrate import cumtrapz, trapz
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter


def read_csv_with_meta(path, ntop=3, nbot=1, sep=",", **kws):
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

    Returns
    -------
    df : pandas.DataFrame
        DataFrame read in
    meta : list
        meta data from top and bottom rows



    """
    with open(path, "r") as f:
        # grap metadata from first rows
        meta = [f.readline().strip() for _ in range(ntop)]
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
    paths : sequence
        collection of paths to read
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
    set_index : bool, default=False
        If True, add `id_vars` and `var_name` to index, if not already there

    append : bool, optional
        if `set_index`, this contols whether new levels are appended to the current
        index, or overwrite the current index.
        Default is to use the opposite of `ignore_index`

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
    groupby=None,
    x="Time [Sec]",
    y=None,
    drop_unused=False,
    reduction=True,
    **kws,
):
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
        if reduction:
            if drop_unused:
                out = df.iloc[[0], :].loc[:, y].copy()
            else:
                out = df.iloc[[0], :].drop(x, axis=1)
        else:
            if drop_unused:
                if x in df.columns:
                    out = df.loc[:, [x] + y].copy()
                else:
                    out = df.loc[:, y].copy()
            else:
                out = df.copy()

        out.loc[:, y] = func(df, x, y, **kws)

        # xvals = _get_col_or_level(df, x).values
        # out.loc[:, y] = trapz(y=df.loc[:, y].values, x=xvals, axis=0)
    else:
        out = pd.concat(
            (
                apply_func_over_groups(
                    func=func,
                    df=g,
                    groupby=None,
                    x=x,
                    y=y,
                    drop_unused=drop_unused,
                    reduction=reduction,
                    **kws,
                )
                for _, g in df.groupby(groupby, sort=False)
            )
        )
    return out


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


def argmin_frame(df, groupby=None, x="Time [Sec]", y=None, drop_unused=False):
    return apply_func_over_groups(
        func=_func_argmin,
        df=df,
        groupby=groupby,
        x=x,
        y=y,
        drop_unused=drop_unused,
        reduction=True,
    )


def argmax_frame(df, groupby=None, x="Time [Sec]", y=None, drop_unused=False):
    return apply_func_over_groups(
        func=_func_argmax,
        df=df,
        groupby=groupby,
        x=x,
        y=y,
        drop_unused=drop_unused,
        reduction=True,
    )


def trapz_frame(df, groupby=None, x="Time [Sec]", y=None, drop_unused=False):
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
    return apply_func_over_groups(
        func=_func_trapz,
        df=df,
        groupby=groupby,
        x=x,
        y=y,
        drop_unused=drop_unused,
        reduction=True,
    )


def cumtrapz_frame(df, groupby=None, x="Time [Sec]", y=None, drop_unused=False):
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
    return apply_func_over_groups(
        func=_func_cumtrapz,
        df=df,
        groupby=groupby,
        x=x,
        y=y,
        drop_unused=drop_unused,
        reduction=False,
    )


def norm_frame(df, groupby=None, x="Time [Sec]", y=None, drop_unused=False):
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
    return apply_func_over_groups(
        func=_func_norm,
        df=df,
        groupby=groupby,
        x=x,
        y=y,
        drop_unused=drop_unused,
        reduction=False,
    )


def gradient_frame(df, groupby=None, x="Time [Sec]", y=None, drop_unused=False, **kws):
    return apply_func_over_groups(
        func=_func_gradient,
        df=df,
        groupby=groupby,
        x=x,
        y=y,
        drop_unused=drop_unused,
        reduction=False,
    )


def savegol_filter_frame(
    df,
    groupby=None,
    x="Time [Sec]",
    y=None,
    drop_unused=False,
    window_length=None,
    polyorder=2,
    **kws,
):

    if window_length is None:
        return df

    kws = dict(window_length=window_length, polyorder=polyorder, **kws)

    return apply_func_over_groups(
        func=_func_savgol_filter,
        df=df,
        groupby=groupby,
        x=x,
        y=y,
        drop_unused=drop_unused,
        reduction=False,
        **kws,
    )


def median_filter_frame(
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

    return apply_func_over_groups(
        func=_func_median_filter,
        df=df,
        groupby=groupby,
        x=x,
        y=y,
        drop_unused=drop_unused,
        reduction=False,
        **kws,
    )


class DataApplier(object):
    def __init__(self, frame, groupby=None, x="Time [Sec]", y=None, drop_unused=False):

        self.frame = frame
        if isinstance(groupby, str):
            groupby = [groupby]
        self.groupby = groupby
        self.x = x
        self.y = y
        self.drop_unused = drop_unused

    def tidy(self, value_name="intensity", var_name="element", id_vars=None, **kws):

        groupby = [] if self.groupby is None else self.groupby
        x = [] if self.x is None else [self.x]

        if id_vars is None:
            id_vars = groupby + x

        data_tidy = tidy_frame(
            self.frame, value_name=value_name, var_name=var_name, id_vars=id_vars, **kws
        )

        return type(self)(
            frame=data_tidy,
            groupby=groupby + [var_name],
            x=self.x,
            y=value_name,
            drop_unused=self.drop_unused,
        )

    def new_like(self, **kws):
        for k in ["frame", "groupby", "x", "y", "drop_unused"]:
            if k not in kws:
                kws[k] = getattr(self, k)
        return type(self)(**kws)

    def _apply_func(
        self,
        _func,
        _reduction=False,
        as_frame=False,
        groupby=None,
        x=None,
        y=None,
        drop_unused=None,
        **kws,
    ):

        if groupby is None:
            groupby = self.groupby
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if drop_unused is None:
            drop_unused = self.drop_unused

        out = _func(
            self.frame, groupby=groupby, x=x, y=y, drop_unused=drop_unused, **kws
        )

        if not as_frame:
            x = None if _reduction else self.x
            out = self.new_like(frame=out, x=x)

        return out

    def trapz(self, **kws):
        return self._apply_func(trapz_frame, True, **kws)

    def cumtrapz(self, **kws):
        return self._apply_func(cumtrapz_frame, **kws)

    def integrate(self, **kws):
        """alias for cumtrapz"""
        return self.cumtrapz(**kws)

    def norm(self, **kws):
        return self._apply_func(norm_frame, **kws)

    def normalize(self, **kws):
        """alieas for norm"""
        return self.norm(**kws)

    def gradient(self, **kws):
        return self._apply_func(gradient_frame, **kws)

    def savgol_filter(self, window_length=None, polyorder=2, **kws):
        return self._apply_func(
            savegol_filter_frame,
            window_length=window_length,
            polyorder=polyorder,
            **kws,
        )

    def median_filter(self, kernel_size=None, **kws):
        return self._apply_func(median_filter_frame, kernel_size=kernel_size, **kws)

    def argmin(self, **kws):
        return self._apply_func(argmin_frame, True, **kws)

    def argmax(self, **kws):
        return self._apply_func(argmax_frame, True, **kws)

    def __repr__(self):
        return "<Applier>\n" + repr(self.frame)

    # def _repr_html_(self):
    #     s = '<title>Applier</title>' + self.frame._repr_html_()
    #     return s

    @property
    def plot(self):
        return self.frame.plot

    def set_index(self, *args, **kwargs):
        return self.new_like(frame=self.frame.set_index(*args, **kwargs))

    @property
    def index(self):
        return self.frame.index

    def get_bounds(
        self, kernel_size=None, is_tidy=True, y=None, mean_over=None, tidy_kws=None
    ):

        data = self
        if kernel_size is not None:
            data = data.median_filter(kernel_size=kernel_size)
        data = data.gradient()

        lb = data.argmax()
        ub = data.argmin()

        if not is_tidy:
            if tidy_kws is None:
                tidy_kws = {}
            if y is None:
                y = tidy_kws.get("y", "intensity")
            if mean_over is None:
                mean_over = tidy_kws.get("var_name", "element")

            lb = lb.tidy(**tidy_kws)
            ub = ub.tidy(**tidy_kws)

        else:
            if y is None:
                y = self.frame.columns.drop(self.groupby + [self.x])[0]
            if mean_over is None:
                mean_over = self.groupby[-1]

        # mean over stuff
        groupby = [k for k in self.groupby if k != mean_over]
        lb = lb.frame.drop(mean_over, axis=1).groupby(groupby).mean()
        ub = ub.frame.drop(mean_over, axis=1).groupby(groupby).mean()

        # make a frame with lower and upper bounds combined
        df = pd.merge(
            lb.rename(columns={y: "lb"}),
            ub.rename(columns={y: "ub"}),
            left_index=True,
            right_index=True,
        )
        # return df

        baseline = (
            df.assign(type_bound="baseline")
            .assign(lower_bound=0.0)
            .assign(upper_bound=lambda x: x["lb"])[
                ["type_bound", "lower_bound", "upper_bound"]
            ]
        )

        signal = (
            df.assign(type_bound="signal")
            .assign(lower_bound=lambda x: x["lb"])
            .assign(upper_bound=lambda x: x["ub"])[
                ["type_bound", "lower_bound", "upper_bound"]
            ]
        )

        return pd.concat((baseline, signal)).sort_index()


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
