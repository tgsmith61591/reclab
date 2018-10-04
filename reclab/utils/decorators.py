# -*- coding: utf-8 -*-

from __future__ import absolute_import

__all__ = [
    'inherit_function_doc'
]


def inherit_function_doc(parent):
    """Inherit a parent instance function's documentation.

    Parameters
    ----------
    parent : callable
        The parent class from which to inherit the documentation. If the
        parent class does not have the function name in its MRO, this will
        fail.

    Examples
    --------
    >>> class A(object):
    ...     def do_something(self):
    ...         '''Does something'''
    >>>
    >>> class B(A):
    ...     @inherit_function_doc(A)
    ...     def do_something(self):
    ...         pass
    >>>
    >>> print(B().do_something.__doc__)
    Does something
    """
    def doc_wrapper(method):
        func_name = method.__name__
        assert (func_name in dir(
            parent)), '%s.%s is not a method! Cannot inherit documentation' % (
            parent.__name__, func_name)

        # Set the documentation. This only ever happens at the time of class
        # definition, and not every time the method is called.
        method.__doc__ = getattr(parent, func_name).__doc__

        # We don't need another wrapper, we can just return the method as its
        # own method
        return method
    return doc_wrapper
