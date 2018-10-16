# -*- coding: utf-8 -*-
# Auto-generated with bear v0.1.9, (c) Taylor G Smith

from __future__ import print_function, division, absolute_import

import os

from reclab._build_utils import maybe_cythonize_extensions


# DEFINE CONFIG
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    libs = []
    if os.name == 'posix':
        libs.append('m')

    config = Configuration('reclab', parent_package, top_path)

    # These are build utilities that are used as a smoke test for whether we
    # end up building the Cython source correctly (only if the package depends
    # on Cython)
    config.add_subpackage('__check_build')
    config.add_subpackage('__check_build/tests')
    config.add_subpackage('_build_utils')

    # This is where submodules get added:
    config.add_subpackage('collab')
    config.add_subpackage('datasets')
    config.add_subpackage('model_deployment')
    config.add_subpackage('model_selection')
    config.add_subpackage('utils')

    # This is where submodule test dirs are added:
    config.add_subpackage('collab/tests')
    # config.add_subpackage('datasets/tests')  # Added in submodule
    config.add_subpackage('model_deployment/tests')
    config.add_subpackage('model_selection/tests')
    config.add_subpackage('utils/tests')

    # This is where the submodules that need cythonizing go:
    config.add_subpackage('metrics')  # adds its own test submodule

    # And this is where we cythonize
    maybe_cythonize_extensions(top_path, config)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
