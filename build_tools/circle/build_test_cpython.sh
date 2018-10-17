#!/usr/bin/env bash
set -x
set -e

sudo -E apt-get -yq update

# deactivate circleci virtualenv and setup a miniconda env instead
if [[ `type -t deactivate` ]]; then
  deactivate
fi

# Install dependencies with miniconda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
   -O miniconda.sh
chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
export PATH="$MINICONDA_PATH/bin:$PATH"
conda update --yes --quiet conda

# Configure the conda environment and put it in the path using the
# provided versions
conda create -n $CONDA_ENV_NAME --yes --quiet python="${PYTHON_VERSION:-*}" \
  numpy="${NUMPY_VERSION:-*}" scipy cython pytest coverage

source activate $CONDA_ENV_NAME
pip install pybind11
pip install "implicit>=$IMPLICIT_VERSION"
pip install nmslib
pip install annoy
pip install codecov
pip install pytest-cov

# Build and install the package in dev mode
python setup.py develop

# Now run the tests with coverage
pytest -v --durations=20 --cov-config .coveragerc --cov reclab
