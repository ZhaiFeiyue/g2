from setuptools import setup
from torch.utils import cpp_extension
from habana_frameworks.torch.utils.lib_utils import get_include_dir, get_lib_dir
import pybind11
from setuptools import setup, find_packages
from typing import List

with open('g2attn/VERSION', 'r') as fp:
    lines = fp.readlines()
    version = lines[0]

torch_include_dir = get_include_dir()
torch_lib_dir = get_lib_dir()
habana_modules_directory = "/usr/include/habanalabs"
pybind_include_path = pybind11.get_include()

def get_requirements() -> List[str]:
    return []

setup(name='g2attn',
      version=version,
      description='A TPC Optimized Attention Lib for Intel Gaudi2',
      url='https://github.com/ZhaiFeiyue/g2',
      author="Zhai Feiyue, Leo Zhao",
      author_email='feiyue.zhai@intel.com, leo.zhao@intel.com',
      py_modules=['g2attn'],
      ext_modules=[cpp_extension.CppExtension('g2attnlib', ['cpp/attn.cpp'],
            libraries=['habana_pytorch_plugin'],
            library_dirs=[torch_lib_dir])],
      include_dirs=[torch_include_dir,
                    habana_modules_directory,
                    pybind_include_path,
                    ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      zip_safe=False,
      packages=find_packages(exclude=("tests")),
      install_requires=get_requirements(),
      package_data={'g2attn': ['VERSION', 'libcustom_tpc_perf_lib.so']},
      )
