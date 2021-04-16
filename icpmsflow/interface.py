"""
simplified interface for analysis
"""


from .cached_decorators import gcached
from .core import cumtrapz_frame, load_paths, norm_frame, tidy_frame
from .plotbounds import DataExplorerPanel


class ICPMSAnalysis:
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
        return DataExplorerPanel(**kws)

    @property
    def bounds_data(self):
        return self.data_explorer.bounds_data

    @classmethod
    def from_paths(cls, paths, load_kws=None, **kws):
        if load_kws is None:
            load_kws = {}

        df, df_meta = load_paths(paths, **load_kws)
        return cls(data=df, meta=df_meta, **kws)
