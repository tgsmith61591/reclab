#!/bin/bash
# This script is meant to be called by the "after_success" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.

# License: 3-clause BSD

# 02/10/2018 remove due to Travis build issue 6307
# set -e
set +e  # because TRAVIS SUCKS

# remove the .egg-info dir so Mac won't bomb on bdist_wheel cmd (absolute path in SOURCES.txt)
rm -r reclab.egg-info/ || echo "No local .egg cache to remove"

# make a dist folder if not there, then make sure permissions are sufficient
mkdir -p dist
chmod 777 dist
