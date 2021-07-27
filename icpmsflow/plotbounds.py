try:
    from ._plotbounds import DataExplorerCombined

except ImportError:

    class DataExplorerCombined:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                """This feature requires bokeh and holoviews.
                Please install bokeh and holoviews."""
            )
