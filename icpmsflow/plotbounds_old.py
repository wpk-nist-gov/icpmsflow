import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
from bokeh.models import HoverTool
from holoviews import opts, streams


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

    # checkboxes for type of plot
    #     Line_plot = param.Boolean(default=True)
    #     Scatter_plot = param.Boolean(default=True)
    Step_plot = param.Boolean(default=False)  # True)

    # dimesion names
    time_dim = param.String(default="Time [Sec]", precedence=-1)
    batch_dim = param.String(default="batch", precedence=-1)
    element_dim = param.String(default="element", precedence=-1)
    variable_dim = param.String(default="variable", precedence=-1)
    value_dim = param.String(default="value", precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)

        # mapping from plain name to actual dimension name
        self._dim_mapping = {
            k: getattr(self, f"{k}_dim")
            for k in ["time", "batch", "element", "variable", "value"]
        }

        # plot stuff
        self._post_data()

        # hover
        tooltips = [
            ("Element", f"@{{{self.element_dim}}}"),
            ("Time", f"@{{{self.time_dim}}}"),
            ("Value", f"@{{{self.value_dim}}}"),
        ]
        hover = HoverTool(tooltips=tooltips)

        self.plot_dmap = hv.DynamicMap(self.get_plot).opts(
            opts.Scatter(framewise=True, tools=[hover], width=600, height=400),
        )

        # table stuff
        if "bounds_data" not in params:
            self.bounds_data = pd.DataFrame(
                {
                    self.batch_dim: self.param["batch"].objects,
                }
            )

        if self.batch_dim not in self.bounds_data.index.names:
            self.bounds_data = self.bounds_data.set_index(self.batch_dim)

        for k in ["lower", "upper"]:
            if k not in self.bounds_data.columns:
                self.bounds_data = self.bounds_data.assign(**{k: np.nan})

        self.table_dmap = hv.DynamicMap(self.get_table).opts(width=600, height=600)

        # span stuff
        self._blank = hv.Curve([]).opts(active_tools=["box_select"])
        self._boundsx = streams.BoundsX(source=self._blank, boundsx=(0, 0))
        self._boundsx.add_subscriber(self._set_bounds)
        self.span_dmap = hv.DynamicMap(self._update_span)  # , streams=[self._boundsx])

    # bounds stuff
    def set_bounds_span(self):
        times = self.plot_dmap.dimension_values(self.time_dim)
        self._set_bounds((times.min(), times.max()))

    @param.depends("bounds")
    def _update_span(self):
        return hv.VSpan(*self.bounds).opts(color="red", alpha=0.2) * self._blank

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

    # Plot stuff
    def _update_params(self):

        for k in ["batch", "variable", "element"]:
            dim = self._dim_mapping[k]
            val = list(self.data[dim].unique())

            self.param[k].objects = val

            if isinstance(self.param[k].default, list):
                v0 = val
            else:
                v0 = val[0]
            setattr(self, k, v0)

    def _update_dataset(self):
        self._ds = hv.Dataset(
            self.data,
            [self.batch_dim, self.variable_dim, self.element_dim, self.time_dim],
        )

    @param.depends("data", watch=True)
    def _post_data(self):
        self._update_params()
        self._update_dataset()

    @param.depends("batch", "variable", "element", "Step_plot")
    def get_plot(self):
        sub = self._ds.select(
            batch=[self.batch], variable=[self.variable], element=self.element
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

        return scat * line

    # Tabe stuff
    @param.depends("bounds")
    def get_table(self):
        if self.bounds != (0.0, 0.0):
            data = self.bounds_data
            data.loc[self.batch, ["lower", "upper"]] = self.bounds
            self.bounds_data = data
        return hv.Table(self.bounds_data.reset_index())

    @property
    def app(self):
        widget_pane = pn.panel(self.param, widgets={"element": {"size": 10}})
        return pn.Row(
            widget_pane, pn.Column(self.plot_dmap * self.span_dmap, self.table_dmap)
        )


# aside looking at bounds
class DataExplorerPanelMultiCol(param.Parameterized):
    # dataframes
    data = param.DataFrame(precedence=-1)
    bounds_data = param.DataFrame(precedence=-1)

    # main widgets
    batch = param.ObjectSelector()
    variable = param.ObjectSelector()
    element = param.ListSelector(default=[])  # height=100)
    bounds = param.Tuple(default=(0.0, 0.0))
    set_bounds = param.Action(lambda self: self.set_bounds_span())

    # checkboxes for type of plot
    Step_plot = param.Boolean(default=False)  # True)

    # dimesion names
    time_dim = param.String(default="Time [Sec]", precedence=-1)
    batch_dim = param.String(default="batch", precedence=-1)
    element_dim = param.String(default="element", precedence=-1)
    # variable_dim = param.String(default='variable', precedence=-1)
    # value_dim = param.String(default='value', precedence=-1)

    def __init__(self, **params):
        super().__init__(**params)

        # mapping from plain name to actual dimension name
        self._dim_mapping = {
            k: getattr(self, f"{k}_dim") for k in ["time", "batch", "element"]
        }

        # plot stuff
        self._post_data()

        # hover
        tooltips = [
            ("Element", f"@{{{self.element_dim}}}"),
            ("Time", f"@{{{self.time_dim}}}"),
        ]
        hover = HoverTool(tooltips=tooltips)

        self.plot_dmap = hv.DynamicMap(self.get_plot).opts(
            opts.Scatter(framewise=True, tools=[hover], width=600, height=400),
        )

        # table stuff
        if "bounds_data" not in params:
            self.bounds_data = pd.DataFrame(
                {
                    self.batch_dim: self.param["batch"].objects,
                }
            )

        if self.batch_dim not in self.bounds_data.index.names:
            self.bounds_data = self.bounds_data.set_index(self.batch_dim)

        for k in ["lower", "upper"]:
            if k not in self.bounds_data.columns:
                self.bounds_data = self.bounds_data.assign(**{k: np.nan})

        self.table_dmap = hv.DynamicMap(self.get_table).opts(width=600, height=600)

        # span stuff
        self._blank = hv.Curve([]).opts(active_tools=["box_select"])
        self._boundsx = streams.BoundsX(source=self._blank, boundsx=(0, 0))
        self._boundsx.add_subscriber(self._set_bounds)
        self.span_dmap = hv.DynamicMap(self._update_span)  # , streams=[self._boundsx])

    # bounds stuff
    def set_bounds_span(self):
        times = self.plot_dmap.dimension_values(self.time_dim)
        self._set_bounds((times.min(), times.max()))

    @param.depends("bounds")
    def _update_span(self):
        return hv.VSpan(*self.bounds).opts(color="red", alpha=0.2) * self._blank

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

    def _update_variable(self):
        names = list(
            self.data.columns.drop([self.element_dim, self.batch_dim, self.time_dim])
        )
        self.param["variable"].objects = names
        self.variable = names[0]

    # Plot stuff
    def _update_params(self):

        for k in ["batch", "element"]:
            dim = self._dim_mapping[k]

            val = list(self.data[dim].unique())

            self.param[k].objects = val

            if isinstance(self.param[k].default, list):
                v0 = val
            else:
                v0 = val[0]
            setattr(self, k, v0)

    def _update_dataset(self):
        self._ds = hv.Dataset(
            self.data,
            [self.batch_dim, self.element_dim, self.time_dim],
            vdims=self.param.variable.objects,
        )

    @param.depends("data", watch=True)
    def _post_data(self):
        self._update_params()
        self._update_variable()
        self._update_dataset()

    @param.depends("batch", "variable", "element", "Step_plot")
    def get_plot(self):
        sub = self._ds.select(batch=[self.batch], element=self.element)

        scat = (
            sub.to(hv.Scatter, self.time_dim, self.variable, self.element_dim)
            .overlay(self.element_dim)
            .opts(opts.Scatter(size=5))
        )
        line = sub.to(hv.Curve, self.time_dim, self.variable, self.element_dim).overlay(
            self.element_dim
        )

        if self.Step_plot:
            line = line.opts(opts.Curve(interpolation="steps-mid"))
        else:
            line = line.opts(opts.Curve(interpolation="linear"))

        return scat * line

    # Tabe stuff
    @param.depends("bounds")
    def get_table(self):
        if self.bounds != (0.0, 0.0):
            data = self.bounds_data
            data.loc[self.batch, ["lower", "upper"]] = self.bounds
            self.bounds_data = data
        return hv.Table(self.bounds_data.reset_index())

    @property
    def app(self):
        # return pn.Column(pn.Row(self.param, self.plot_dmap * self.span_dmap), pn.Pane(self.table_dmap))

        widget_pane = pn.panel(self.param, widgets={"element": {"size": 10}})

        return pn.Row(
            widget_pane, pn.Column(self.plot_dmap * self.span_dmap, self.table_dmap)
        )
