"""

Core routines to analyze icpms data

"""

from datetime import datetime

# import numpy as np
import pandas as pd

# from pathlib import Path
from scipy.integrate import cumtrapz, trapz


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
    **kws
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
        **kws
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


from scipy.ndimage import median_filter


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
                    **kws
                )
                for _, g in df.groupby(groupby, sort=False)
            )
        )
    return out


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
