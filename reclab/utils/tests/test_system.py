# -*- coding: utf-8 -*-

from __future__ import absolute_import

from reclab.utils.system import safe_mkdirs
import shutil


def test_mkdirs():
    loc = "here"
    try:
        safe_mkdirs(loc)

        # Show we can make it again and it doesn't break down
        safe_mkdirs(loc)
    finally:
        shutil.rmtree(loc)
