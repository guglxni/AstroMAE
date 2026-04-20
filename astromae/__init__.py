"""AstroMAE: Masked Autoencoders for astrophysical images."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("astromae")
except PackageNotFoundError:
    __version__ = "0.2.0.dev0"

__all__ = ["__version__"]
