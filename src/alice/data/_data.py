"""
# -*- coding: utf-8 -*-
# Copyright © 2020 Abhishek Pratapa. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
"""

from abc import ABC, abstractmethod

class DataTemplate(ABC):
    @abstractmethod
    def has_next(self):
        pass

    @abstractmethod
    def next(self, has_label=True):
        pass
