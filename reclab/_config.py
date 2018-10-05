# -*- coding: utf-8 -*-

from __future__ import absolute_import

import os
from os.path import expanduser

# The directory in which we'll store NMS & Annoy model indices for pickling
RECLAB_CACHE = os.environ.get('RECLAB_CACHE',
                              expanduser('~/.reclab-cache'))
