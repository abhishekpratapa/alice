"""
# -*- coding: utf-8 -*-
# Copyright © 2020 Abhishek Pratapa. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
"""

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure


class Cache:
    def __init__(self, ticker):
        self.client = MongoClient('mongodb://localhost:27017/')
        self.ticker = ticker
        self.db = None
        self.collection = None
        self.connected = False
        self.__test_connection()
        self.__init_connection()

    def __test_connection(self):
        try:
            self.client.server_info()
            self.connected = True
        except ConnectionFailure:
            pass

    def __init_connection(self):
        if self.connected:
            self.db = self.client.stocks
            self.collection = self.db[self.ticker]

    def get_bars(self, current_day):
        if self.connected:
            current_bars = self.collection.find_one({"date": current_day})
            if current_bars != None:
                return current_bars['data']

        return None

    def add_bars(self, current_day, bars):
        if self.connected:
            bar = dict()
            bar["date"] = current_day
            bar["data"] = bars
            self.collection.insert_one(bar)