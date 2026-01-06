from pathlib import Path
__root_path__ = str(Path(__file__).absolute().parent)
from ._get_version import __version__
from ._plot_cfg import *
from .utils import *
from .nn import *

del Path
