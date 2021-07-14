import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.models import HoverTool
from holoviews import opts, streams

from .core import DataApplier

_VERBOSE = True


def _vprint(*args):
    if _VERBOSE:
        print(*args)


# class EditableDataFrame(param.Parameterized):

#     batch = param.ObjectSelector(default='a', objects=['a','b'])
#     bounds_data = param.


class SpanExplorer(param.Parameterized):
    """
    object to handle vertical spans
    """

    batch = param.ObjectSelector(default="a", objects=["a", "b"])

    bounds = param.Tuple(default=(0.0, 0.0), precedence=-1)

    # baseline_lower = param.Number(step=0.1)
    # baseline_upper = param.Number(step=0.1)
    # signal_lower = param.Number(step=0.1)
    # signal_upper = param.Number(step=0.1)

    bounds_type = param.Selector(
        default="baseline", objects=["baseline", "signal"], label="Bounds Name"
    )
    cycle_bounds = param.Boolean(default=True)

    clear_bounds = param.Action(lambda self: self._clear_bounds())
    clear_all_bounds = param.Action(lambda self: self._clear_all_bounds())

    bounds_data = param.DataFrame(precedence=-1)
    batch_dim = param.String(default="batch", precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)
        self.set_bounds_data(self.bounds_data)
        self._init_span()

    @property
    def _bounds_limit_names(self):
        return ["lower_bound", "upper_bound"]

    @property
    def _bounds_type_name(self):
        return "type_bound"

    @property
    def bounds_dim(self):
        return [self.batch_dim, self._bounds_type_name]

    # bounds
    # @param.depends("bounds_data", watch=True)
    def set_bounds_data(self, bounds_data):
        if bounds_data is None:
            bounds_data = pd.DataFrame(
                columns=self.bounds_dim + self._bounds_limit_names
            )

        missing = [b for b in self.bounds_dim if b not in bounds_data.index.names]
        if set(missing) == set(self.bounds_dim):
            bounds_data = bounds_data.set_index(self.bounds_dim)
        elif len(missing) > 0:
            bounds_data = bounds_data.set_index(missing, append=True)

        # reorder
        try:
            bounds_data.index = bounds_data.index.reorder_levels(self.bounds_dim)
        except Exception:
            pass
        self.bounds_data = bounds_data
        self._table_dmap = hv.DynamicMap(self.get_table).opts(width=600, height=300)

    # def _update_bounds_to_table(self, bounds, cycle):
    #     kws = {}
    #     if bounds != (0.0, 0.0):
    #         data = self.bounds_data
    #         data.loc[
    #             (self.batch, self.bounds_type), self._bounds_limit_names
    #         ] = bounds
    #         self.bounds_data = data

    #     if cycle:
    #         objects = self.param.bounds_type.objects
    #         index = objects.index(self.bounds_type)
    #         self.bounds_type = objects[(index + 1) % len(objects)]

    @param.depends("bounds", watch=True)
    def _add_bounds_to_table(self):
        if self.bounds != (0.0, 0.0):
            data = self.bounds_data
            data.loc[
                (self.batch, self.bounds_type), self._bounds_limit_names
            ] = self.bounds
            self.bounds_data = data

        # if self.bounds_type == "signal":
        #     self.bounds_signal = self.bounds
        # else:
        #     self.bounds_baseline = self.bounds

        if self.cycle_bounds:
            objects = self.param.bounds_type.objects
            index = objects.index(self.bounds_type)
            self.bounds_type = objects[(index + 1) % len(objects)]

    # @param.depends('bounds_signal', watch=True)
    # def _set_bounds_from_bounds_signal(self):
    #     # self._calls.append('signal')
    #     self.param.set_param(bounds_type='signal', bounds=self.bounds_signal)

    # @param.depends('bounds_baseline', watch=True)
    # def _set_bounds_from_bounds_baseline(self):
    #     # self._calls.append('baseline')
    #     self.param.set_param(bounds_type='baseline', bounds=self.bounds_baseline)

    # @param.depends('bounds','bounds_type', watch=True)
    # def _set_from_bounds(self):
    #     # self._calls.append('bounds')
    #     if self.bounds_type == 'signal':
    #         self.bounds_signal = self.bounds
    #     else:
    #         self.bounds_baseline = self.bounds

    @param.depends("bounds_data")
    def get_table(self):
        return hv.Table(self.bounds_data.reset_index())

    def _init_span(self):
        self._blank = hv.Curve([]).opts(active_tools=["box_select"])
        self._boundsx = streams.BoundsX(source=self._blank, boundsx=(0, 0))
        self._boundsx.add_subscriber(self._set_bounds)
        self._span_dmap = hv.DynamicMap(self._update_span)

    @param.depends("bounds_data", "batch")
    def _update_span(self):
        span = []
        data = self.bounds_data.query(f"{self.batch_dim} == @self.batch")
        for i, g in data.reset_index().iterrows():
            if g[self._bounds_type_name] == "baseline":
                color = "blue"
            else:
                color = "red"

            lower, upper = g.loc[self._bounds_limit_names]

            span.append(hv.VSpan(lower, upper).opts(color=color, alpha=0.2))
        if len(span) == 0:
            span = [self._blank]
        return hv.Overlay(span)

    @param.depends("batch", watch=True)
    def _reset_bounds_type_to_default(self):
        """reset bounds type after batch select"""
        self.bounds_type = self.param.bounds_type.default

    def _set_bounds(self, boundsx):
        self.bounds = tuple(round(x, 3) for x in boundsx)

    # @param.depends("batch", watch=True)
    # def _update_bounds_after_batch(self):
    #     try:

    #         bounds = tuple(
    #             self.bounds_data.loc[(self.batch, self.bounds_type), ["lower", "upper"]].fillna(0.0).values
    #         )
    #     except Exception:
    #         bounds = (0.0, 0.0)
    #     self.bounds = bounds

    def _clear_bounds(self):
        # placeholder for clearing bounds of currently selected type
        q = f"{self.batch_dim} != @self.batch or {self._bounds_type_name} != @self.bounds_type"
        self.bounds_data = self.bounds_data.query(q)

    def _clear_all_bounds(self):
        q = f"{self.batch_dim} != @self.batch"
        self.bounds_data = self.bounds_data.query(q)

    def get_plot_dmap(self):
        return pn.Column(self._span_dmap * self._blank, self._table_dmap)

    @property
    def widgets(self):
        return pn.Param(
            self.param,
            name="Bounds",
            widgets={"bounds_type": pn.widgets.RadioButtonGroup},
        )

    @property
    def app(self):
        return pn.Row(self.widgets, self.get_plot_dmap)


class SpanExplorerInput(param.Parameterized):
    """
    object to handle vertical spans
    """

    batch = param.ObjectSelector(default="a", objects=["a", "b", "c"])

    bounds = param.Tuple(default=(0.0, 0.0), precedence=-1)

    baseline_lower = param.Number(step=0.1, bounds=(0.0, None))
    baseline_upper = param.Number(step=0.1, bounds=(0.0, None))
    signal_lower = param.Number(step=0.1, bounds=(0.0, None))
    signal_upper = param.Number(step=0.1, bounds=(0.0, None))

    bounds_type = param.Selector(
        default="baseline", objects=["baseline", "signal"], label="Bounds Name"
    )
    cycle_bounds = param.Boolean(default=True)

    clear_bounds = param.Action(lambda self: self._clear_bounds())
    clear_all_bounds = param.Action(lambda self: self._clear_all_bounds())

    bounds_data = param.DataFrame(precedence=-1)
    batch_dim = param.String(default="batch", precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)
        self.set_bounds_data(self.bounds_data)
        self._init_span()

    @property
    def _bounds_limit_names(self):
        return ["lower_bound", "upper_bound"]

    @property
    def _bounds_type_name(self):
        return "type_bound"

    @property
    def bounds_dim(self):
        return [self.batch_dim, self._bounds_type_name]

    def set_bounds_data(self, bounds_data):
        if bounds_data is None:
            bounds_data = pd.DataFrame(
                columns=self.bounds_dim + self._bounds_limit_names
            )

        missing = [b for b in self.bounds_dim if b not in bounds_data.index.names]
        if set(missing) == set(self.bounds_dim):
            bounds_data = bounds_data.set_index(self.bounds_dim)
        elif len(missing) > 0:
            bounds_data = bounds_data.set_index(missing, append=True)

        # reorder
        try:
            bounds_data.index = bounds_data.index.reorder_levels(self.bounds_dim)
        except Exception:
            pass
        self.bounds_data = bounds_data
        self._table_dmap = hv.DynamicMap(self.get_table).opts(width=600, height=300)

    def _add_bounds_to_table(self, bounds, bounds_type):
        # Top level function to add bounds to table
        if bounds != (0.0, 0.0) and bounds[0] <= bounds[1]:
            data = self.bounds_data
            data.loc[(self.batch, bounds_type), self._bounds_limit_names] = bounds
            self.bounds_data = data

    @param.depends("bounds", watch=True)
    def _update_lower_upper(self):
        d = {}
        if self.bounds_type == "baseline":
            d["baseline_lower"], d["baseline_upper"] = self.bounds
        elif self.bounds_type == "signal":
            d["signal_lower"], d["signal_upper"] = self.bounds

        self.param.set_param(**d)

        if self.cycle_bounds:
            objects = self.param.bounds_type.objects
            index = objects.index(self.bounds_type)
            self.bounds_type = objects[(index + 1) % len(objects)]

    @param.depends("baseline_lower", "baseline_upper", watch=True)
    def _update_table_baseline(self):
        bounds = (self.baseline_lower, self.baseline_upper)
        self._add_bounds_to_table(bounds, "baseline")

    @param.depends("signal_lower", "signal_upper", watch=True)
    def _update_table_signal(self):
        bounds = (self.signal_lower, self.signal_upper)
        self._add_bounds_to_table(bounds, "signal")

    @param.depends("bounds_data")
    def get_table(self):
        return hv.Table(self.bounds_data.reset_index())

    def _init_span(self):
        self._blank = hv.Curve([]).opts(active_tools=["box_select"])
        self._boundsx = streams.BoundsX(source=self._blank, boundsx=(0, 0))
        self._boundsx.add_subscriber(self._set_bounds)
        self._span_dmap = hv.DynamicMap(self._update_span)

    @param.depends("bounds_data", "batch")
    def _update_span(self):
        span = []
        data = self.bounds_data.query(f"{self.batch_dim} == @self.batch")
        for i, g in data.reset_index().iterrows():
            if g[self._bounds_type_name] == "baseline":
                color = "blue"
            else:
                color = "red"

            lower, upper = g.loc[self._bounds_limit_names]

            span.append(hv.VSpan(lower, upper).opts(color=color, alpha=0.2))
        if len(span) == 0:
            span = [self._blank]
        return hv.Overlay(span)

    @param.depends("batch", watch=True)
    def _reset_bounds_type_to_default(self):
        """reset bounds type after batch select"""
        if self.cycle_bounds:
            self.bounds_type = self.param.bounds_type.default

        # update bounds if available:
        bounds = {}
        for name in ["baseline", "signal"]:
            try:
                bounds[name] = tuple(
                    self.bounds_data.loc[(self.batch, name), self._bounds_limit_names]
                )
            except Exception:
                bounds[name] = (0.0, 0.0)
        self._safe_bounds = bounds

        self.param.set_param(
            baseline_lower=bounds["baseline"][0],
            baseline_upper=bounds["baseline"][1],
            signal_lower=bounds["signal"][0],
            signal_upper=bounds["signal"][1],
        )

    def _set_bounds(self, boundsx):
        self.bounds = tuple(round(x, 3) for x in boundsx)

    def _clear_bounds(self):
        # placeholder for clearing bounds of currently selected type
        q = f"{self.batch_dim} != @self.batch or {self._bounds_type_name} != @self.bounds_type"
        self.bounds_data = self.bounds_data.query(q)

    def _clear_all_bounds(self):
        q = f"{self.batch_dim} != @self.batch"
        self.bounds_data = self.bounds_data.query(q)

    def get_plot_dmap(self):
        return pn.Column(self._span_dmap * self._blank, self._table_dmap)

    @property
    def widgets(self):
        return pn.Param(
            self.param,
            name="Bounds",
            widgets={"bounds_type": pn.widgets.RadioButtonGroup},
        )

    @property
    def app(self):
        return pn.Row(self.widgets, self.get_plot_dmap)


class SpanExplorerTabulator0(param.Parameterized):
    """
    object to handle vertical spans
    """

    batch = param.ObjectSelector(default="a", objects=list("abcdefghijk"))

    bounds = param.Tuple(default=(0.0, 0.0), precedence=-1)

    bounds_type = param.Selector(
        default="baseline", objects=["baseline", "signal"], label="Bounds Name"
    )
    cycle_bounds = param.Boolean(default=True)

    clear_bounds = param.Action(lambda self: self._clear_bounds())
    clear_all_bounds = param.Action(lambda self: self._clear_all_bounds())

    bounds_data = param.DataFrame(precedence=-1)
    batch_dim = param.String(default="batch", precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)
        self.set_bounds_data(self.bounds_data)
        self._init_span()

    def set_bounds_data(self, bounds_data):
        if bounds_data is None:
            bounds_data = pd.DataFrame(
                columns=self.bounds_dim + self._bounds_limit_names
            )

        if isinstance(bounds_data.index, pd.MultiIndex):
            bounds_data = bounds_data.reset_index()

        order = self.bounds_dim + self._bounds_limit_names
        bounds_data = (
            bounds_data[order]
            .set_index(self.bounds_dim)
            .reindex(
                pd.MultiIndex.from_product(
                    (self.param.batch.objects, self.param.bounds_type.objects),
                    names=self.bounds_dim,
                )
            )
            .reset_index()
        )

        # make sure is numeric
        bounds_data = bounds_data.applymap(lambda x: pd.to_numeric(x, errors="ignore"))
        self.bounds_data = bounds_data
        editors = {
            k: {"type": "number", "step": 0.5, "min": 0.0}
            for k in self._bounds_limit_names
        }

        self.table = pn.widgets.Tabulator(
            bounds_data, show_index=False, editors=editors
        )  # self.bokeh_editors)#, show_index=False)

        self._reset_bounds_type_to_default()

    @property
    def _bounds_limit_names(self):
        return ["lower_bound", "upper_bound"]

    @property
    def _bounds_type_name(self):
        return "type_bound"

    #     @property
    #     def _pivot_names(self):
    #         return [f'{a}:{b}' for a in x.param.bounds_type.objects for b in x._bounds_limit_names]

    #     @property
    #     def _baseline_signal

    @property
    def bounds_dim(self):
        return [self.batch_dim, self._bounds_type_name]

    @param.depends("bounds", watch=True)
    def _add_bounds_to_table(self):
        if not hasattr(self, "table"):
            return

        if self.bounds != (0.0, 0.0):

            data = self.table.value.set_index(self.bounds_dim)
            data.loc[
                (self.batch, self.bounds_type), self._bounds_limit_names
            ] = self.bounds
            self.table.value = data.reset_index()

        if self.cycle_bounds:
            objects = self.param.bounds_type.objects
            index = objects.index(self.bounds_type)
            self.bounds_type = objects[(index + 1) % len(objects)]

    # @param.depends('table.param', "batch")
    def _get_table(self):
        # print("_get_table")
        # self._reset_table_selection()
        return self.table

    def get_table(self):
        return pn.Pane(self._get_table)

    @param.depends("batch", watch=True)
    def _reset_table_selection(self):
        print("reset_table_select")
        if not hasattr(self, "table"):
            return
        data = self.table.value
        iloc = list(np.where(data.batch == self.batch)[0])

        kws = {}
        if tuple(self.table.frozen_rows) != tuple(iloc):
            kws["frozen_rows"] = iloc
        if tuple(self.table.selection) != tuple(iloc):
            kws["selection"] = iloc

        if len(kws) > 0:
            self.table.param.set_param(**kws)

    def _init_span(self):
        self._blank = hv.Curve([]).opts(active_tools=["box_select"])
        self._boundsx = streams.BoundsX(source=self._blank, boundsx=(0, 0))
        self._boundsx.add_subscriber(self._set_bounds)
        self._span_dmap = hv.DynamicMap(self._update_span)

    @param.depends("table.value", "batch")
    def _update_span(self):
        print("update_span")
        if not hasattr(self, "table"):
            return
        span = []
        data = self.table.value.query(f"{self.batch_dim} == @self.batch").fillna(0.0)
        for i, g in data.reset_index().iterrows():
            if g[self._bounds_type_name] == "baseline":
                color = "blue"
            else:
                color = "red"

            lower, upper = g.loc[self._bounds_limit_names]
            if upper >= lower:
                span.append(hv.VSpan(lower, upper).opts(color=color, alpha=0.2))

        self._current_samp = span
        if len(span) == 0:
            span = [self._blank]
        return hv.Overlay(span)

    @param.depends("batch", watch=True)
    def _reset_bounds_type_to_default(self):
        print("reset_bounds_type_to_default")
        """reset bounds type after batch select"""
        #         print('in reset batch')
        self.bounds_type = self.param.bounds_type.default

    #         self._reset_table_selection()

    def _set_bounds(self, boundsx):
        self.bounds = tuple(round(x, 3) for x in boundsx)

    def _clear_bounds(self):
        # placeholder for clearing bounds of currently selected type
        data = self.table.value
        msk = (data[self.batch_dim] == self.batch) & (
            data[self._bounds_type_name] == self.bounds_type
        )
        data.loc[msk, self._bounds_limit_names] = np.nan

        self.table.value = data

    #         self._reset_table_selection()

    def _clear_all_bounds(self):
        data = self.table.value
        msk = data[self.batch_dim] == self.batch
        data.loc[msk, self._bounds_limit_names] = np.nan

        self.table.value = data

    #         self._reset_table_selection()

    def get_plot_dmap(self):
        return pn.Column(self._span_dmap * self._blank, self.get_table)

    @property
    def widgets(self):
        return pn.Param(
            self.param,
            name="Bounds",
            widgets={"bounds_type": pn.widgets.RadioButtonGroup},
        )

    @property
    def app(self):
        return pn.Row(self.widgets, self.get_plot_dmap)


import time
from functools import wraps


# def timing_val(func):
#     def wrapper(*arg, **kw):
#         '''source: http://www.daniweb.com/code/snippet368.html'''
#         t1 = time.time()
#         res = func(*arg, **kw)
#         t2 = time.time()
#         return (t2 - t1), res, func.__name__
#     return wrapper
def verbose(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        t0 = time.time()
        out = func(self, *args, **kwargs)
        t1 = time.time()
        print(f"{func.__name__}: {t1 - t0}")
        return out

    return wrapper


class SpanExplorerTabulator(param.Parameterized):
    """
    object to handle vertical spans
    """

    batch = param.ObjectSelector(default="a", objects=list("abcdefghijk"))

    bounds = param.Tuple(default=(0.0, 0.0), precedence=-1)

    bounds_type = param.Selector(
        default="baseline", objects=["baseline", "signal"], label="Bounds Name"
    )
    cycle_bounds = param.Boolean(default=True)

    clear_bounds = param.Action(lambda self: self._clear_bounds())
    clear_all_bounds = param.Action(lambda self: self._clear_all_bounds())

    bounds_data = param.DataFrame(precedence=-1)
    batch_dim = param.String(default="batch", precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)
        self.set_bounds_data(self.bounds_data)
        self._init_span()

    @verbose
    def set_bounds_data(self, bounds_data):
        if bounds_data is None:
            bounds_data = pd.DataFrame(
                columns=self.bounds_dim + self._bounds_limit_names
            )

        if isinstance(bounds_data.index, pd.MultiIndex):
            bounds_data = bounds_data.reset_index()

        order = self.bounds_dim + self._bounds_limit_names
        bounds_data = (
            bounds_data[order]
            .set_index(self.bounds_dim)
            .reindex(
                pd.MultiIndex.from_product(
                    (self.param.batch.objects, self.param.bounds_type.objects),
                    names=self.bounds_dim,
                )
            )
            .reset_index()
        )

        # make sure is numeric
        bounds_data = bounds_data.applymap(lambda x: pd.to_numeric(x, errors="ignore"))
        self.bounds_data = bounds_data
        editors = {
            k: {"type": "number", "step": 0.5, "min": 0.0}
            for k in self._bounds_limit_names
        }

        self.table = pn.widgets.Tabulator(
            bounds_data, show_index=False, editors=editors
        )  # self.bokeh_editors)#, show_index=False)

        self._reset_bounds_type_to_default()

    @property
    def _bounds_limit_names(self):
        return ["lower_bound", "upper_bound"]

    @property
    def _bounds_type_name(self):
        return "type_bound"

    #     @property
    #     def _pivot_names(self):
    #         return [f'{a}:{b}' for a in x.param.bounds_type.objects for b in x._bounds_limit_names]

    #     @property
    #     def _baseline_signal

    @property
    def bounds_dim(self):
        return [self.batch_dim, self._bounds_type_name]

    @verbose
    @param.depends("bounds", watch=True)
    def _add_bounds_to_table(self):
        if not hasattr(self, "table"):
            return
        if self.bounds != (0.0, 0.0):

            data = self.table.value.set_index(self.bounds_dim)
            data.loc[
                (self.batch, self.bounds_type), self._bounds_limit_names
            ] = self.bounds
            self.table.value = data.reset_index()

        if self.cycle_bounds:
            objects = self.param.bounds_type.objects
            index = objects.index(self.bounds_type)
            self.bounds_type = objects[(index + 1) % len(objects)]

    @verbose
    @param.depends("batch", watch=True)
    def _reset_table_selection(self):
        if not hasattr(self, "table"):
            return
        data = self.table.value
        iloc = list(np.where(data.batch == self.batch)[0])

        kws = {}
        if tuple(self.table.frozen_rows) != tuple(iloc):
            kws["frozen_rows"] = iloc
        if tuple(self.table.selection) != tuple(iloc):
            kws["selection"] = iloc

        if len(kws) > 0:
            self.table.param.set_param(**kws)

    @verbose
    def _init_span(self):
        self._blank = hv.Curve([]).opts(active_tools=["box_select"])
        self._boundsx = streams.BoundsX(source=self._blank, boundsx=(0, 0))
        self._boundsx.add_subscriber(self._set_bounds)
        self._span_dmap = hv.DynamicMap(self._update_span)

    @verbose
    @param.depends("table.value", "batch")
    def _update_span(self):
        if not hasattr(self, "table"):
            return
        span = []
        data = self.table.value.query(f"{self.batch_dim} == @self.batch").fillna(0.0)
        for i, g in data.reset_index().iterrows():
            if g[self._bounds_type_name] == "baseline":
                color = "blue"
            else:
                color = "red"

            lower, upper = g.loc[self._bounds_limit_names]
            if upper >= lower:
                span.append(hv.VSpan(lower, upper).opts(color=color, alpha=0.2))

        self._current_samp = span
        if len(span) == 0:
            span = [self._blank]
        return hv.Overlay(span)

    @verbose
    @param.depends("batch", watch=True)
    def _reset_bounds_type_to_default(self):
        """reset bounds type after batch select"""
        self.bounds_type = self.param.bounds_type.default

    #         self._reset_table_selection()

    def _set_bounds(self, boundsx):
        self.bounds = tuple(round(x, 3) for x in boundsx)

    @verbose
    def _clear_bounds(self):

        # placeholder for clearing bounds of currently selected type
        data = self.table.value
        # q = f"{self.batch_dim} == @self.batch and {self._bounds_type_name} == @self.bounds_type"
        msk = (data[self.batch_dim] == self.batch) & (
            data[self._bounds_type_name] == self.bounds_type
        )
        data.loc[msk, self._bounds_limit_names] = np.nan

        self.table.value = data

    #         self._reset_table_selection()

    @verbose
    def _clear_all_bounds(self):
        data = self.table.value
        msk = data[self.batch_dim] == self.batch
        data.loc[msk, self._bounds_limit_names] = np.nan

        self.table.value = data

    #         self._reset_table_selection()

    def get_plot_dmap(self):
        return pn.Column(self._span_dmap * self._blank, self.table)  # get_table)

    @property
    def widgets(self):
        return pn.Param(
            self.param,
            name="Bounds",
            widgets={"bounds_type": pn.widgets.RadioButtonGroup},
        )

    @property
    def app(self):
        return pn.Row(self.widgets, self.get_plot_dmap)


class PlotExplorer(param.Parameterized):
    # dataframes
    data = param.DataFrame(precedence=-1)

    # main widgets
    batch = param.ObjectSelector()
    variable = param.ObjectSelector()
    element = param.ListSelector(default=[])

    # checkboxes for type of plot
    integrate_check = param.Boolean(default=False, label="Integrate")
    normalize_check = param.Boolean(default=False, label="Normalize")
    gradient_check = param.Boolean(default=False, label="Gradient")
    smooth_line_check = param.Boolean(default=False, label="Median filter (line)")
    smooth_scat_check = param.Boolean(default=False, label="Meidan filter (scatter)")
    smooth_param = param.Integer(default=3, bounds=(3, 101), label="Filter width")

    step_plot = param.Boolean(default=False)  # True)

    # dimesion names
    time_dim = param.String(default="Time [Sec]", precedence=-1)
    batch_dim = param.String(default="batch", precedence=-1)
    element_dim = param.String(default="element", precedence=-1)
    variable_dim = param.String(default="variable", precedence=-1)
    value_dim = param.String(default="value", precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)

        # mapping from local names to actual name
        self._dim_mapping = {
            k: getattr(self, f"{k}_dim")
            for k in ["time", "batch", "element", "variable", "value"]
        }

        if self.data is not None:
            self._post_data()

    @verbose
    def _update_options(self):
        """update options, and returns dict of values to be batch set"""

        if self.data is not None:
            query = []
            values_dict = {}
            for k in ["batch", "variable", "element"]:
                dim = self._dim_mapping[k]

                if dim not in self.data.columns:
                    self.param[k].precedence = -1
                else:
                    self.param[k].precedence = None

                    val = list(self.data[dim].unique())
                    self.param[k].objects = val

                    if isinstance(self.param[k].default, list):
                        v0 = val
                        q = f"{dim} in @self.{k}"
                    else:
                        v0 = val[0]
                        q = f"{dim} == @self.{k}"
                    values_dict[k] = v0
                    query.append(q)

            self._query = " and ".join(query)
        else:
            values_dict = None

        return values_dict

    @property
    def kdims(self):
        return [
            k
            for k in [self.batch_dim, self.variable_dim, self.element_dim]
            if k in self.data.columns
        ]

    @verbose
    def _set_plot_dmap(self):
        tooltips = [
            ("Element", f"@{{{self.element_dim}}}"),
            ("Time", f"@{{{self.time_dim}}}"),
            ("Value", f"@{{{self.value_dim}}}"),
        ]
        hover = HoverTool(tooltips=tooltips)

        self._plot_dmap = hv.DynamicMap(self.get_plot).opts(
            opts.Scatter(framewise=True, tools=[hover], width=600, height=400),
        )

    @verbose
    @param.depends("data", watch=True)
    def _post_data(self):
        values_dict = self._update_options()
        # self._update_dataset()
        self._set_plot_dmap()
        self.param.set_param(**values_dict)
        # setup plot_dmap

    @verbose
    @param.depends("data")
    def get_plot_dmap(self):
        if hasattr(self, "_plot_dmap"):
            plot = self._plot_dmap  # self._table_dmap# * self._span_dmap
        else:
            plot = hv.Curve([])

        return plot

    @verbose
    def _apply_all_funcs(self, sub, names=None):

        if names is None:
            names = ["integrate", "normalize", "smooth_line", "smooth_scat", "gradient"]

        checks = {k: getattr(self, k + "_check") for k in names}

        # limit names
        names = [k for k, v in checks.items() if v]

        # limit names to only those cases where
        if names:

            d = DataApplier(
                sub,
                by=self.kdims,
                x_dim=self.time_dim,
                y_dim=self.value_dim,
            )

            # do integration and normalization first
            for name in ["integrate", "normalize"]:
                if checks[name]:
                    d = getattr(d, name)()

            # do smoothing before gradient

            if checks["smooth_line"] or checks["smooth_scat"]:
                smoothed = d.median_filter(kernel_size=self.smooth_param)

                d_line = smoothed if checks["smooth_line"] else d
                d_scat = smoothed if checks["smooth_scat"] else d

            else:
                d_line = d_scat = d

            if checks["gradient"]:
                if d_line is d_scat:
                    d_line = d_scat = d_line.gradient()
                else:
                    d_line, d_scat = [x.gradient() for x in [d_line, d_scat]]

            sub_line, sub_scat = d_line.frame, d_scat.frame

        else:
            sub_line = sub_scat = sub

        return sub_line, sub_scat

    def _to_dataset(self, sub_line, sub_scat):

        kdims = self.kdims
        vdims = [self.value_dim]

        if sub_line is sub_scat:
            ds_line = ds_scat = hv.Dataset(sub_line, kdims=kdims, vdims=vdims)
        else:
            ds_line = hv.Dataset(sub_line, kdims, vdims)
            ds_scat = hv.Dataset(sub_scat, kdims, vdims)
        return ds_line, ds_scat

    @verbose
    @param.depends(
        "batch",
        "variable",
        "element",
        "step_plot",
        "normalize_check",
        "integrate_check",
        "gradient_check",
        "smooth_line_check",
        "smooth_scat_check",
        "smooth_param",
    )
    def get_plot(self):
        if self.data is not None:

            sub = self.data.query(self._query)

            # apply functions
            sub_line, sub_scat = self._apply_all_funcs(sub)

            ds_line, ds_scat = self._to_dataset(sub_line, sub_scat)

            scat = (
                ds_scat.to(hv.Scatter, self.time_dim, self.value_dim, self.element_dim)
                .overlay(self.element_dim)
                .opts(opts.Scatter(size=5, alpha=1.0))
            )
            line = ds_line.to(
                hv.Curve, self.time_dim, self.value_dim, self.element_dim
            ).overlay(self.element_dim)

            if self.step_plot:
                line = line.opts(opts.Curve(interpolation="steps-mid"))
            else:
                line = line.opts(opts.Curve(interpolation="linear"))

        else:
            scat = hv.Scatter([])
            line = hv.Curve([])

        return scat * line

    # @property
    # def widgets(self):
    #     return pn.Param(self.param, name='Data Selector')

    @property
    def app(self):
        return pn.Row(self.param, self.get_plot_dmap)


class DataExplorerCombined(SpanExplorerInput, PlotExplorer):
    def __init__(self, **params):

        super().__init__(**params)

        # mapping from local names to actual name
        self._dim_mapping = {
            k: getattr(self, f"{k}_dim")
            for k in ["time", "batch", "element", "variable", "value"]
        }

        # if self.data is not None:
        #     self._post_data()

        self.set_bounds_data(self.bounds_data)
        self._init_span()

    # specialization for this object
    @param.depends("data")
    def get_plot_dmap(self):
        if hasattr(self, "_plot_dmap"):
            plot = (
                self._plot_dmap * self._blank * self._span_dmap
            )  # self._table_dmap# * self._span_dmap
        else:
            plot = hv.Curve([])
        return pn.Column(plot, self._table_dmap)

    @property
    def widgets(self):
        return pn.Param(
            self.param,
            name="Data Explorer",
            widgets={"bounds_type": pn.widgets.RadioButtonGroup},
        )

    @property
    def app(self):
        return pn.Row(self.widgets, self.get_plot_dmap)


class DataExplorerCombinedTabulator(SpanExplorerTabulator, PlotExplorer):
    def __init__(self, **params):

        super().__init__(**params)

        # mapping from local names to actual name
        self._dim_mapping = {
            k: getattr(self, f"{k}_dim")
            for k in ["time", "batch", "element", "variable", "value"]
        }

        # _vprint('pre')
        # if self.data is not None:
        #     self._post_data()

        # _vprint('post')

        self.set_bounds_data(self.bounds_data)
        self._init_span()

    # specialization for this object
    @verbose
    @param.depends("data")
    def get_plot_dmap(self):
        if hasattr(self, "_plot_dmap"):
            plot = (
                self._plot_dmap * self._blank * self._span_dmap
            )  # self._table_dmap# * self._span_dmap
        else:
            plot = hv.Curve([])

        # return plot
        return pn.Column(plot, pn.Pane(self.table))

    @property
    def widgets(self):
        return pn.Param(
            self.param,
            name="Data Explorer",
            widgets={"bounds_type": pn.widgets.RadioButtonGroup},
        )

    @property
    def app(self):
        return pn.Row(
            self.widgets, self.get_plot_dmap
        )  # pn.Column(self.get_plot_dmap, self.get_table))


# class DataExplorerPanel(param.Parameterized):
#     # dataframes
#     data = param.DataFrame(precedence=-1)
#     bounds_data = param.DataFrame(precedence=-1)

#     # main widgets
#     batch = param.ObjectSelector()
#     variable = param.ObjectSelector()
#     element = param.ListSelector(default=[])  # height=100)

#     bounds = param.Tuple(default=(0.0, 0.0))

#     set_bounds = param.Action(lambda self: self.set_bounds_span())
#     clear_bounds = param.Action(lambda self: self._clear_bounds())

#     # checkboxes for type of plot
#     Step_plot = param.Boolean(default=False)  # True)

#     # dimesion names
#     time_dim = param.String(default="Time [Sec]", precedence=-1)
#     batch_dim = param.String(default="batch", precedence=-1)
#     element_dim = param.String(default="element", precedence=-1)
#     variable_dim = param.String(default="variable", precedence=-1)
#     value_dim = param.String(default="value", precedence=-1)

#     def __init__(self, **params):
#         super().__init__(**params)

#         # mapping from local names to actual name
#         self._dim_mapping = {
#             k: getattr(self, f"{k}_dim")
#             for k in ["time", "batch", "element", "variable", "value"]
#         }

#         if self.data is not None:
#             self._post_data()

#         self._init_bounds_data()
#         self._init_span()

#     def _update_options(self):
#         """update options, and returns dict of values to be batch set"""
#         if self.data is not None:
#             values_dict = {}
#             for k in ["batch", "variable", "element"]:
#                 _vprint("setting", k)
#                 dim = self._dim_mapping[k]
#                 val = list(self.data[dim].unique())
#                 self.param[k].objects = val

#                 if isinstance(self.param[k].default, list):
#                     v0 = val
#                 else:
#                     v0 = val[0]
#                 values_dict[k] = v0
#         else:
#             values_dict = None

#         return values_dict

#     def _update_dataset(self):
#         _vprint("update dataset")
#         if self.data is not None:
#             self._ds = hv.Dataset(
#                 self.data,
#                 [self.batch_dim, self.variable_dim, self.element_dim, self.time_dim],
#             )
#         else:
#             self._ds = None

#     def _set_plot_dmap(self):
#         tooltips = [
#             ("Element", f"@{{{self.element_dim}}}"),
#             ("Time", f"@{{{self.time_dim}}}"),
#             ("Value", f"@{{{self.value_dim}}}"),
#         ]
#         hover = HoverTool(tooltips=tooltips)

#         self._plot_dmap = hv.DynamicMap(self.get_plot).opts(
#             opts.Scatter(framewise=True, tools=[hover], width=600, height=400),
#         )

#     @param.depends("data", watch=True)
#     def _post_data(self):
#         _vprint("post data")
#         values_dict = self._update_options()
#         self._update_dataset()

#         self._set_plot_dmap()
#         self.param.set_param(**values_dict)
#         # setup plot_dmap

#     @param.depends("data")
#     def get_plot_dmap(self):
#         if hasattr(self, "_plot_dmap"):
#             plot = (
#                 self._plot_dmap * self._blank * self._span_dmap
#             )  # self._table_dmap# * self._span_dmap
#         else:
#             plot = hv.Curve([])

#         return pn.Column(plot, self._table_dmap)

#     # bounds
#     def _init_bounds_data(self):
#         if self.bounds_data is None:
#             self.bounds_data = pd.DataFrame(columns=[self.batch_dim, "lower", "upper"])
#         if self.batch_dim not in self.bounds_data.index.names:
#             self.bounds_data = self.bounds_data.set_index(self.batch_dim)
#         self._table_dmap = hv.DynamicMap(self.get_table).opts(width=600, height=600)

#     @param.depends("bounds")
#     def get_table(self):
#         if self.bounds != (0.0, 0.0):
#             data = self.bounds_data
#             data.loc[self.batch, ["lower", "upper"]] = self.bounds
#             self.bounds_data = data
#         return hv.Table(self.bounds_data.reset_index())

#     def _init_span(self):
#         self._blank = hv.Curve([]).opts(active_tools=["box_select"])
#         self._boundsx = streams.BoundsX(source=self._blank, boundsx=(0, 0))
#         self._boundsx.add_subscriber(self._set_bounds)
#         self._span_dmap = hv.DynamicMap(self._update_span)

#     @param.depends("bounds")
#     def _update_span(self):
#         return hv.VSpan(*self.bounds).opts(color="red", alpha=0.2)  # * self._blank

#     def _set_bounds(self, boundsx):
#         self.bounds = tuple(round(x, 3) for x in boundsx)

#     @param.depends("batch", watch=True)
#     def _update_bounds_after_batch(self):
#         try:
#             bounds = tuple(
#                 self.bounds_data.loc[self.batch, ["lower", "upper"]].fillna(0.0).values
#             )
#         except Exception:
#             bounds = (0.0, 0.0)
#         self.bounds = bounds

#     def set_bounds_span(self):
#         times = self._plot_dmap.dimension_values(self.time_dim)
#         self._set_bounds((times.min(), times.max()))

#     def _clear_bounds(self):
#         self.bounds_data = self.bounds_data.query(f'{self.batch_dim} != "{self.batch}"')

#     # @param.depends("batch", "variable", "element", "Step_plot")
#     # def get_plot(self):
#     #     _vprint("updating plot")
#     #     if self.data is not None:
#     #         sub = self._ds.select(
#     #             batch=[self.batch], variable=[self.variable], element=self.element
#     #         )
#     #         scat = (
#     #             sub.to(hv.Scatter, self.time_dim, self.value_dim, self.element_dim)
#     #             .overlay(self.element_dim)
#     #             .opts(opts.Scatter(size=5))
#     #         )
#     #         line = sub.to(
#     #             hv.Curve, self.time_dim, self.value_dim, self.element_dim
#     #         ).overlay(self.element_dim)

#     #         if self.Step_plot:
#     #             line = line.opts(opts.Curve(interpolation="steps-mid"))
#     #         else:
#     #             line = line.opts(opts.Curve(interpolation="linear"))

#     #     else:
#     #         scat = hv.Scatter([])
#     #         line = hv.Curve([])

#     #     return scat * line
#     @param.depends("batch", "variable", "element", "Step_plot")
#     def get_plot(self):
#         _vprint("updating plot")
#         if self.data is not None:

#             q = f"{self.batch_dim}==@self.batch and {self.variable_dim}==@self.variable and {self.element_dim} in @self.element"
#             sub = self.data.query(q)

#             sub = hv.Dataset(
#                 sub,
#                 [self.batch_dim, self.variable_dim, self.element_dim, self.time_dim],
#                 [self.value_dim],
#             )

#             scat = (
#                 sub.to(hv.Scatter, self.time_dim, self.value_dim, self.element_dim)
#                 .overlay(self.element_dim)
#                 .opts(opts.Scatter(size=5))
#             )
#             line = sub.to(
#                 hv.Curve, self.time_dim, self.value_dim, self.element_dim
#             ).overlay(self.element_dim)

#             if self.Step_plot:
#                 line = line.opts(opts.Curve(interpolation="steps-mid"))
#             else:
#                 line = line.opts(opts.Curve(interpolation="linear"))

#         else:
#             scat = hv.Scatter([])
#             line = hv.Curve([])

#         return scat * line

#     @property
#     def app(self):
#         return pn.Row(self.param, self.get_plot_dmap)
