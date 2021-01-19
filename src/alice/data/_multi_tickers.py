"""
# -*- coding: utf-8 -*-
# Copyright © 2020 Abhishek Pratapa. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
"""

import random

from ._data import DataTemplate
from ._stock_trader import StockDataLoader
from ..labeler import SellLabeler


class MultiTicker(DataTemplate):
    def __init__(self, tickers, config):
        self.tickers = tickers
        self.config = config
        self.loaders = dict()
        self.__populate_tickers()

    def __populate_tickers(self):
        for tick in self.tickers:
            try:
                temp_loader = self.__process_tick(tick)
                if len(temp_loader['loader'].date) > 0:
                    self.loaders[tick] = temp_loader
                    print("Loading: " + tick)
                else:
                    raise
            except:
                print("Passing: " + tick)

    def __process_tick(self, tick):
        labeler = SellLabeler(
            self.config.timestep,
            self.config.num_bins,
            self.config.step_size
        )

        loader = StockDataLoader(
            tick,
            self.config.batch_size,
            self.config.start_date,
            self.config.end_date,
            self.config.min_sequence,
            self.config.max_sequence,
            self.config.rolling_window,
            self.config.num_bins,
            self.config.alpaca_key,
            self.config.alpaca_secrets,
            labeler
        )

        return {
            'labeler': labeler,
            'loader': loader
        }

    def __random_loader(self):
        loader_keys = list(self.loaders.keys())
        selected_key = random.choice(loader_keys)
        return self.loaders[selected_key]['loader']

    def has_next(self):
        return len(list(self.loaders.keys())) > 0

    def next(self, has_label=True):
        current_loader = self.__random_loader()
        return current_loader.next()
