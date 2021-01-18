"""
# -*- coding: utf-8 -*-
# Copyright © 2020 Abhishek Pratapa. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
"""

from pymongo import MongoClient

class Cache:
    def __init__(self, ticker):
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client.stocks
        self.ticker = ticker
        self.collection = self.db[ticker]

    def get_bars(self, current_day):
        current_bars = self.collection.find_one({"date": current_day})
        return current_bars['data']

    def add_bars(self, current_day, bars):
        bar = dict()
        bar["date"] = current_day
        bar["data"] = bars
        self.collection.insert_one(bar)