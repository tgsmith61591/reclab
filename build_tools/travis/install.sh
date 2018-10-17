#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

# Install pip requirements
travis_wait travis_retry $PIP install numpy cython scikit-learn pybind11
travis_wait travis retry $PIP install "implicit>=0.3.7" nmslib annoy pytest twine

# now run the python setup. This implicitly builds all the C code with build_ext
travis_retry $PIP install -e .
