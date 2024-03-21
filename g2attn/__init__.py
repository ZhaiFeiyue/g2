import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
import torch
import habana_frameworks.torch.core

module_dir = os.path.realpath(__file__)
module_dir = os.path.dirname(module_dir)

ver_file = os.path.join(module_dir, 'VERSION')
with open(ver_file, 'r') as fp:
        lines = fp.readlines()

__version__ = lines[0]

so_dir = Path(module_dir).parent
g2attnlib_path = [str(i) for i in so_dir.glob("g2attnlib.cpython*-x86_64-linux-gnu.so")]

assert len(g2attnlib_path) != 0, 'not found g2attnlib'
torch.ops.load_library(g2attnlib_path[0])
