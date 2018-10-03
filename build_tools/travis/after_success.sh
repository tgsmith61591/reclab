#!/bin/bash
# This script is meant to be called by the "after_success" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.

# License: 3-clause BSD

# 02/10/2018 remove due to Travis build issue 6307
# set -e
set +e  # because TRAVIS SUCKS

# push coverage if necessary
if [[ "$COVERAGE" == "true" ]]; then

    # Need to run coveralls from a git checkout, so we copy .coverage
    # from TEST_DIR where nosetests has been run
    cp $TEST_DIR/.coverage $TRAVIS_BUILD_DIR
    cd $TRAVIS_BUILD_DIR

    # Ignore codecov failures as the codecov server is not
    # very reliable but we don't want travis to report a failure
    # in the github UI just because the coverage report failed to
    # be published.
    codecov || echo "codecov upload failed"
fi

# make sure we have twine in case we deploy
pip install twine || "pip installing twine failed"

# remove the .egg-info dir so Mac won't bomb on bdist_wheel cmd (absolute path in SOURCES.txt)
rm -r reclab.egg-info/ || echo "No local .egg cache to remove"

# make a dist folder if not there, then make sure permissions are sufficient
mkdir -p dist
chmod 777 dist
