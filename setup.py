#!/usr/bin/env python

"""PyMKS - the materials knowledge system in Python

See the documenation for details at https://pymks.org
"""

<<<<<<< HEAD
import pathlib
=======
import configparser
import pathlib
import warnings
import os
import subprocess
from distutils.util import strtobool

from setuptools import setup, find_packages, Extension
from setuptools.config import read_configuration
>>>>>>> 66d44020b9da0a2b8e684257eb12ff4d6c9ff32c

from setuptools.config import read_configuration
from setuptools import setup, find_packages
import versioneer


def get_setupcfg():
    """Get the absolute path for setup.cfg
    """
    return pathlib.Path(__file__).parent.absolute() / "setup.cfg"


def get_configuration():
    """Get contents of setup.cfg as a dict
    """

    return read_configuration(get_setupcfg())


def get_name():
    """Single location for name of package
    """
    return get_configuration()["metadata"]["name"]


<<<<<<< HEAD
=======
def get_setupcfg():
    """Get the absolute path for setup.cfg
    """
    return pathlib.Path(__file__).parent.absolute() / "setup.cfg"


def get_configuration():
    """Get contents of setup.cfg as a dict
    """

    return read_configuration(get_setupcfg())


def read_all_config(option):
    """Read all of the options in setup.cfg not just those used in setup.
    """
    parser = configparser.ConfigParser()
    parser.read(get_setupcfg())
    if parser.has_option("pymks", option):
        return parser.getboolean("pymks", option)
    return False


def env_var(var):
    """Determine the value of an enviroment variable

    Args:
      var: variable

    Returns:
      (defined, value): `defined` is bool depending on whether
        var is defined, `value` is the bool value it's set
        to (positive if undetermined)
    """
    defined = var in os.environ
    if defined:
        env_string = os.environ[var]
        try:
            value = strtobool(env_string)
        except ValueError:
            value = True
    else:
        value = False
    return defined, value


def get_name():
    """Single location for name of package
    """
    return get_configuration()["metadata"]["name"]


def build_graspi():
    """Decide whether to build Graspi
    """
    env_defined, env_value = env_var("PYMKS_USE_BOOST")
    if env_defined:
        return env_value
    return read_all_config("use-boost")


def graspi_path():
    """Find the path to graspi
    """
    return list(filter(lambda x: "graspi" in x, find_packages()))[0].replace(".", "/")


def graspi_extension():
    """Configure the graspi extension

    """
    import numpy  # pylint: disable=import-outside-toplevel

    return Extension(
        name=graspi_path().replace("/", ".") + ".graspi",
        sources=[
            os.path.join(graspi_path(), "graspi.pyx"),
            os.path.join(graspi_path(), "graph_constructors.cpp"),
        ],
        include_dirs=[numpy.get_include(), graspi_path(), "."],
        extra_compile_args=["-std=c++11"],
        language="c++",
        optional=True,
    )


def get_extensions():
    """Get all extensions, return empty dict if no extension modules
    activated

    """

    def cythonize(*args, **kwargs):
        """Only import cython if actually using cython to build.
        """
        from Cython.Build import (  # pylint: disable=import-outside-toplevel
            cythonize as cythonize_,
        )

        return cythonize_(*args, **kwargs)

    return (
        dict(
            ext_modules=cythonize(
                [graspi_extension()],
                compiler_directives={"language_level": "3"},
                include_path=[graspi_path()],
            )
        )
        if build_graspi()
        else dict()
    )


>>>>>>> 66d44020b9da0a2b8e684257eb12ff4d6c9ff32c
def setup_args():
    """Get the setup arguments not configured in setup.cfg
    """
    return dict(
<<<<<<< HEAD
        packages=find_packages(),
        package_data={"": ["tests/*.py"]},
        data_files=["setup.cfg"],
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
=======
        version=make_version(get_name()),
        packages=find_packages(),
        package_data={"": ["tests/*.py"]},
        data_files=["setup.cfg"],
        **get_extensions()
>>>>>>> 66d44020b9da0a2b8e684257eb12ff4d6c9ff32c
    )


setup(**setup_args())
