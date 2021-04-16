from .core import cumtrapz_frame, load_paths, norm_frame, tidy_frame
from .interface import ICPMSAnalysis
from .plotbounds import DataExplorerPanel

# Version info
try:
    import pkg_resources

    __version__ = pkg_resources.get_distribution("cmomy").version
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

__author__ = """William P. Krekelberg"""
__email__ = "wpk@nist.gov"


__all__ = [
    "load_paths",
    "cumtrapz_frame",
    "norm_frame",
    "tidy_frame",
    "DataExplorerPanel",
    "ICPMSAnalysis",
]
