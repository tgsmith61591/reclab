# -*- coding: utf-8 -*-

from __future__ import absolute_import

from reclab.utils.decorators import inherit_function_doc

import pytest


# Internal class used several times
class A(object):
    def do_something(self):
        """Does something"""


class TestInheritDoc:
    def test_inherit_doc(self):
        class B(A):
            @inherit_function_doc(A)
            def do_something(self):
                pass

        assert B().do_something.__doc__ == "Does something", \
            B().do_something.__doc__

    def test_inherit_doc_assertion_error(self):
        with pytest.raises(AssertionError):
            class C(A):
                @inherit_function_doc(A)
                def do_something_else(self):
                    # Fails since this method is not in the MRO
                    pass
