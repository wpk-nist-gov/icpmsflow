from .core import DataApplier, load_paths  # cumtrapz_frame, norm_frame, tidy_frame
from .interface import ICPMSAnalysis
from .plotbounds import DataExplorerCombined

# Version info
try:
    import pkg_resources

    __version__ = pkg_resources.get_distribution("icpmsflow").version
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

__author__ = """William P. Krekelberg"""
__email__ = "wpk@nist.gov"


__all__ = [
    "DataApplier",
    "load_paths",
    # "cumtrapz_frame",
    # "norm_frame",
    # "tidy_frame",
    "DataExplorerCombined",
    "ICPMSAnalysis",
]
