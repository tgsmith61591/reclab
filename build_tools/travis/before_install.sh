#!/usr/bin/env bash

# remove due to Travis build issue 6307
# set -e
set +e  # because Travis can't get its act together

# if it's a linux build, we need to apt-get update
if [[ "$TRAVIS_OS_NAME" == "osx" ]]; then
  brew update
  brew install gcc
  brew upgrade python
  brew install python3

  export PIP=pip3
  export PY=python3

else
  echo "Travis builds for reclab are currently only intended for Mac OSX"
  exit 4
fi
