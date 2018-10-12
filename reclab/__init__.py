# -*- coding: utf-8 -*-
# Auto-generated with bear v0.1.9, (c) Taylor G Smith


__version__ = '0.0.5'

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of bear when
    # the binaries are not built
    __RECLAB_SETUP__
except NameError:
    __RECLAB_SETUP__ = False

if __RECLAB_SETUP__:
    import sys as _sys
    _sys.stdout.write('Partial import of reclab during '
                      'the build process.\n')
    del _sys
else:
    from . import __check_build
    # Global namespace imports
    # Here, you'll import submodules from your package to make them importable
    # at the top level of the package. For instance, if you want to be able to
    # import 'reclab.utils', you'd mark 'utils' here as follows:
    __all__ = [
        'collab',
        'metrics',
        'model_selection',
        'utils'
    ]


# function for finding the package
def package_location():
    import os
    return os.path.abspath(os.path.dirname(__file__))


def setup_module(module):
    # Fixture to assure global seeding of RNG
    import numpy as np
    import random

    _random_seed = int(np.random.uniform() * (2 ** 31 - 1))
    np.random.seed(_random_seed)
    random.seed(_random_seed)
