import holoviews as hv

# import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.models import HoverTool
from holoviews import opts, streams

from .core import cumtrapz_frame, median_filter_frame, norm_frame


class SpanExplorer(param.Parameterized):
    """
    object to handle vertical spans
    """

    batch = param.ObjectSelector(default="a", objects=["a", "b"])

    bounds = param.Tuple(default=(0.0, 0.0), precedence=-1)
    bounds_type = param.Selector(default="signal", objects=["signal", "baseline"])

    clear_bounds = param.Action(lambda self: self._clear_bounds())
    clear_all_bounds = param.Action(lambda self: self._clear_all_bounds())

    bounds_data = param.DataFrame(precedence=-1)
    batch_dim = param.String(default="batch", precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)
        self._init_bounds_data()
        self._init_span()

    @property
    def bounds_dim(self):
        return [self.batch_dim, "type"]

    # bounds
    def _init_bounds_data(self):
        bounds_data = self.bounds_data
        if bounds_data is None:
            bounds_data = pd.DataFrame(columns=self.bounds_dim + ["lower", "upper"])

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
        self._table_dmap = hv.DynamicMap(self.get_table).opts(width=600, height=600)

    @param.depends("bounds", watch=True)
    def _add_bounds_to_table(self):
        if self.bounds != (0.0, 0.0):
            data = self.bounds_data
            data.loc[(self.batch, self.bounds_type), ["lower", "upper"]] = self.bounds
            self.bounds_data = data

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
            if g["type"] == "baseline":
                color = "red"
            else:
                color = "blue"

            span.append(hv.VSpan(g.lower, g.upper).opts(color=color, alpha=0.2))
        if len(span) == 0:
            span = [self._blank]
        return hv.Overlay(span)

    @param.depends("batch", watch=True)
    def _reset_bounds_type_to_default(self):
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
        q = f"{self.batch_dim} != @self.batch or type != @self.bounds_type"
        self.bounds_data = self.bounds_data.query(q)

    def _clear_all_bounds(self):
        q = f"{self.batch_dim} != @self.batch"
        self.bounds_data = self.bounds_data.query(q)


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

    def _update_options(self):
        """update options, and returns dict of values to be batch set"""
        if self.data is not None:
            values_dict = {}
            for k in ["batch", "variable", "element"]:
                print("setting", k)
                dim = self._dim_mapping[k]
                val = list(self.data[dim].unique())
                self.param[k].objects = val

                if isinstance(self.param[k].default, list):
                    v0 = val
                else:
                    v0 = val[0]
                values_dict[k] = v0
        else:
            values_dict = None

        return values_dict

    # def _update_dataset(self):
    #     print("update dataset")
    #     if self.data is not None:
    #         self._ds = hv.Dataset(
    #             self.data,
    #             [self.batch_dim, self.variable_dim, self.element_dim, self.time_dim],
    #         )
    #     else:
    #         self._ds = None

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

    @param.depends("data", watch=True)
    def _post_data(self):
        print("post data")
        values_dict = self._update_options()
        # self._update_dataset()

        self._set_plot_dmap()
        self.param.set_param(**values_dict)
        # setup plot_dmap

    @param.depends("data")
    def get_plot_dmap(self):
        if hasattr(self, "_plot_dmap"):
            plot = self._plot_dmap  # self._table_dmap# * self._span_dmap
        else:
            plot = hv.Curve([])

        return plot

    # @param.depends("batch", "variable", "element", "Step_plot")
    # def get_plot(self):
    #     print("updating plot")
    #     if self.data is not None:
    #         sub = self._ds.select(
    #             batch=[self.batch], variable=[self.variable], element=self.element
    #         )
    #         scat = (
    #             sub.to(hv.Scatter, self.time_dim, self.value_dim, self.element_dim)
    #             .overlay(self.element_dim)
    #             .opts(opts.Scatter(size=5))
    #         )
    #         line = sub.to(
    #             hv.Curve, self.time_dim, self.value_dim, self.element_dim
    #         ).overlay(self.element_dim)

    #         if self.Step_plot:
    #             line = line.opts(opts.Curve(interpolation="steps-mid"))
    #         else:
    #             line = line.opts(opts.Curve(interpolation="linear"))

    #     else:
    #         scat = hv.Scatter([])
    #         line = hv.Curve([])

    #     return scat * line

    def _apply_func(self, func, sub, groupby="default", x=None, y=None, **kws):
        if groupby == "default":
            groupby = [self.batch_dim, self.element_dim, self.variable_dim]
        if x is None:
            x = self.time_dim
        if y is None:
            y = self.value_dim
        return func(sub, groupby=groupby, x=x, y=y, **kws)

    # def _cumtrap(self, sub, **kws):
    #     return self._apply_func(cumtrapz_frame, sub, **kws)

    # def _norm(self, sub, **kws):
    #     return self._apply_func(norm_frame, sub, **kws)

    # def _smooth(self, sub, **kws):
    #     return self._apply_func(median_filter_frame, sub, **kws)

    def _apply_all_funcs(self, sub, names=None):

        if names is None:
            names = ["integrate", "normalize"]
        checks = [getattr(self, name + "_check") for name in names]
        # precs = [getattr(self, name + '_prec') for name in names]
        # names = [
        #     name
        #     for prec, name, check in
        #     sorted(zip(precs, names, checks), key=lambda x: x[0])
        #     if check
        # ]

        names = [name for name, check in zip(names, checks) if check]

        for name in names:
            if name == "integrate":
                sub = self._apply_func(cumtrapz_frame, sub)

            elif name == "normalize":
                sub = self._apply_func(norm_frame, sub)

            elif name == "smooth":
                sub = self._apply_func(
                    median_filter_frame, sub, kernel_size=self.smooth_param
                )
        return sub

    def _apply_smooth(self, sub):
        if self.smooth_line_check or self.smooth_scat_check:
            smoothed = self._apply_func(
                median_filter_frame, sub, kernel_size=self.smooth_param
            )
            sub_line = smoothed if self.smooth_line_check else sub
            sub_scat = smoothed if self.smooth_scat_check else sub

        else:
            sub_line = sub_scat = sub
        return sub_line, sub_scat

    def _to_dataset(self, sub_line, sub_scat):

        kdims = [self.batch_dim, self.variable_dim, self.element_dim, self.time_dim]
        vdims = [self.value_dim]

        if sub_line is sub_scat:
            ds_line = ds_scat = hv.Dataset(sub_line, kdims=kdims, vdims=vdims)
        else:
            ds_line = hv.Dataset(sub_line, kdims, vdims)
            ds_scat = hv.Dataset(sub_scat, kdims, vdims)
        return ds_line, ds_scat

    @param.depends(
        "batch",
        "variable",
        "element",
        "step_plot",
        "normalize_check",
        "integrate_check",
        "smooth_line_check",
        "smooth_scat_check",
        "smooth_param",
    )  # , 'integrate_prec','normalize_prec','smooth_prec')
    def get_plot(self):
        print("updating plot")
        if self.data is not None:

            sub = self.data.query(
                "batch==@self.batch and variable==@self.variable and element in @self.element"
            )

            # apply functions
            sub = self._apply_all_funcs(sub)

            sub_line, sub_scat = self._apply_smooth(sub)

            ds_line, ds_scat = self._to_dataset(sub_line, sub_scat)

            # sub = hv.Dataset(sub, [self.batch_dim, self.variable_dim, self.element_dim, self.time_dim], [self.value_dim])

            scat = (
                ds_scat.to(hv.Scatter, self.time_dim, self.value_dim, self.element_dim)
                .overlay(self.element_dim)
                .opts(opts.Scatter(size=5))
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

    @property
    def app(self):
        return pn.Row(self.param, self.get_plot_dmap)


class DataExplorerCombined(SpanExplorer, PlotExplorer):
    def __init__(self, **params):

        super().__init__(**params)

        # mapping from local names to actual name
        self._dim_mapping = {
            k: getattr(self, f"{k}_dim")
            for k in ["time", "batch", "element", "variable", "value"]
        }

        # print('pre')

        # if self.data is not None:
        #     self._post_data()

        # print('post')

        self._init_bounds_data()
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
    def app(self):
        return pn.Row(self.param, self.get_plot_dmap)


class DataExplorerPanel(param.Parameterized):
    # dataframes
    data = param.DataFrame(precedence=-1)
    bounds_data = param.DataFrame(precedence=-1)

    # main widgets
    batch = param.ObjectSelector()
    variable = param.ObjectSelector()
    element = param.ListSelector(default=[])  # height=100)

    bounds = param.Tuple(default=(0.0, 0.0))

    set_bounds = param.Action(lambda self: self.set_bounds_span())
    clear_bounds = param.Action(lambda self: self._clear_bounds())

    # checkboxes for type of plot
    Step_plot = param.Boolean(default=False)  # True)

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

        self._init_bounds_data()
        self._init_span()

    def _update_options(self):
        """update options, and returns dict of values to be batch set"""
        if self.data is not None:
            values_dict = {}
            for k in ["batch", "variable", "element"]:
                print("setting", k)
                dim = self._dim_mapping[k]
                val = list(self.data[dim].unique())
                self.param[k].objects = val

                if isinstance(self.param[k].default, list):
                    v0 = val
                else:
                    v0 = val[0]
                values_dict[k] = v0
        else:
            values_dict = None

        return values_dict

    def _update_dataset(self):
        print("update dataset")
        if self.data is not None:
            self._ds = hv.Dataset(
                self.data,
                [self.batch_dim, self.variable_dim, self.element_dim, self.time_dim],
            )
        else:
            self._ds = None

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

    @param.depends("data", watch=True)
    def _post_data(self):
        print("post data")
        values_dict = self._update_options()
        self._update_dataset()

        self._set_plot_dmap()
        self.param.set_param(**values_dict)
        # setup plot_dmap

    @param.depends("data")
    def get_plot_dmap(self):
        if hasattr(self, "_plot_dmap"):
            plot = (
                self._plot_dmap * self._blank * self._span_dmap
            )  # self._table_dmap# * self._span_dmap
        else:
            plot = hv.Curve([])

        return pn.Column(plot, self._table_dmap)

    # bounds
    def _init_bounds_data(self):
        if self.bounds_data is None:
            self.bounds_data = pd.DataFrame(columns=[self.batch_dim, "lower", "upper"])
        if self.batch_dim not in self.bounds_data.index.names:
            self.bounds_data = self.bounds_data.set_index(self.batch_dim)
        self._table_dmap = hv.DynamicMap(self.get_table).opts(width=600, height=600)

    @param.depends("bounds")
    def get_table(self):
        if self.bounds != (0.0, 0.0):
            data = self.bounds_data
            data.loc[self.batch, ["lower", "upper"]] = self.bounds
            self.bounds_data = data
        return hv.Table(self.bounds_data.reset_index())

    def _init_span(self):
        self._blank = hv.Curve([]).opts(active_tools=["box_select"])
        self._boundsx = streams.BoundsX(source=self._blank, boundsx=(0, 0))
        self._boundsx.add_subscriber(self._set_bounds)
        self._span_dmap = hv.DynamicMap(self._update_span)

    @param.depends("bounds")
    def _update_span(self):
        return hv.VSpan(*self.bounds).opts(color="red", alpha=0.2)  # * self._blank

    def _set_bounds(self, boundsx):
        self.bounds = tuple(round(x, 3) for x in boundsx)

    @param.depends("batch", watch=True)
    def _update_bounds_after_batch(self):
        try:
            bounds = tuple(
                self.bounds_data.loc[self.batch, ["lower", "upper"]].fillna(0.0).values
            )
        except Exception:
            bounds = (0.0, 0.0)
        self.bounds = bounds

    def set_bounds_span(self):
        times = self._plot_dmap.dimension_values(self.time_dim)
        self._set_bounds((times.min(), times.max()))

    def _clear_bounds(self):
        self.bounds_data = self.bounds_data.query(f'{self.batch_dim} != "{self.batch}"')

    # @param.depends("batch", "variable", "element", "Step_plot")
    # def get_plot(self):
    #     print("updating plot")
    #     if self.data is not None:
    #         sub = self._ds.select(
    #             batch=[self.batch], variable=[self.variable], element=self.element
    #         )
    #         scat = (
    #             sub.to(hv.Scatter, self.time_dim, self.value_dim, self.element_dim)
    #             .overlay(self.element_dim)
    #             .opts(opts.Scatter(size=5))
    #         )
    #         line = sub.to(
    #             hv.Curve, self.time_dim, self.value_dim, self.element_dim
    #         ).overlay(self.element_dim)

    #         if self.Step_plot:
    #             line = line.opts(opts.Curve(interpolation="steps-mid"))
    #         else:
    #             line = line.opts(opts.Curve(interpolation="linear"))

    #     else:
    #         scat = hv.Scatter([])
    #         line = hv.Curve([])

    #     return scat * line
    @param.depends("batch", "variable", "element", "Step_plot")
    def get_plot(self):
        print("updating plot")
        if self.data is not None:

            q = f"{self.batch_dim}==@self.batch and {self.variable_dim}==@self.variable and {self.element_dim} in @self.element"
            sub = self.data.query(q)

            sub = hv.Dataset(
                sub,
                [self.batch_dim, self.variable_dim, self.element_dim, self.time_dim],
                [self.value_dim],
            )

            scat = (
                sub.to(hv.Scatter, self.time_dim, self.value_dim, self.element_dim)
                .overlay(self.element_dim)
                .opts(opts.Scatter(size=5))
            )
            line = sub.to(
                hv.Curve, self.time_dim, self.value_dim, self.element_dim
            ).overlay(self.element_dim)

            if self.Step_plot:
                line = line.opts(opts.Curve(interpolation="steps-mid"))
            else:
                line = line.opts(opts.Curve(interpolation="linear"))

        else:
            scat = hv.Scatter([])
            line = hv.Curve([])

        return scat * line

    @property
    def app(self):
        return pn.Row(self.param, self.get_plot_dmap)
