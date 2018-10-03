# Building the documentation

You'll need several packages for this:

* sphinx
* sphinx_gallery
* numpydoc

To build the documentation, follow these steps:

```bash
$ python setup.py install
$ cd doc
$ make clean html EXAMPLES_PATTERN=ex_*
```

The examples pattern (`ex_*`) is one that can change depending on your naming
convention. The examples are located in the `examples/` directory, and each
subdirectory should be named according to the submodule it represents.
