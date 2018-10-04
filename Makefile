# simple makefile to simplify repetitive build env management tasks under posix
# this is adopted from the sklearn Makefile. This is REQUIRED for deployment via
# Travis using the build_many_wheels.sh script

# caution: testing won't work on windows

PYTHON ?= python
CYTHON ?= cython
CTAGS ?= ctags

# skip doctests on 32bit python
BITS := $(shell python -c 'import struct; print(8 * struct.calcsize("P"))')

all: clean inplace #test

clean-ctags:
	rm -f tags

clean: clean-ctags
	$(PYTHON) setup.py clean
	rm -rf dist

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

trailing-spaces:
	find python -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) --python-kinds=-i -R reclab

code-analysis:
	flake8 reclab | grep -v __init__ | grep -v external
	pylint -E -i y reclab/ -d E1103,E0611,E1101
