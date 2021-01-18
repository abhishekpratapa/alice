"""
# -*- coding: utf-8 -*-
# Copyright © 2020 Abhishek Pratapa. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
"""

from abc import ABC, abstractmethod

class ModelTemplate(ABC):
    @abstractmethod
    def train(self, inputs, labels, weights):
        pass

    @abstractmethod
    def predict(self, inputs):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def identifier(self):
        pass