"""
# -*- coding: utf-8 -*-
# Copyright © 2020 Abhishek Pratapa. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
"""

import alpaca_trade_api as tradeapi
import datetime
import numpy as np
from pymongo import MongoClient
import pytz
import time
import random

from ._data import DataTemplate
from ..cache import Cache

class StockDataLoader(DataTemplate):
    futura = 1
    def __init__(self, ticker, batch_size, start, end, min_sequence, max_sequence, rolling_window, num_bins, alpaca_key, alpaca_secret, labeler=None):
        self.labeler = labeler
        self.ticker = ticker
        self.batch_size = batch_size
        self.cursor = 0
        self.start = start
        self.end = end
        self.cache = Cache(ticker)
        self.min_sequence = min_sequence
        self.max_sequence = max_sequence
        self.num_bins = num_bins
        self.rolling_window = rolling_window
        self.api = tradeapi.REST(alpaca_key, alpaca_secret, api_version='v2')
        self.data = dict()
        self.label = dict()
        self.date = []
        self.__populate_data()
        self.__populate_labels()
        self.__reset()

    def __populate_labels(self):
        if self.labeler:
            while self.__cursor_next():
                # TODO: hit mongodb database
                date, data = self.__sequential(False)
                self.label[date] = self.labeler.process_data(data)

    def __populate_data(self):
        delta = datetime.timedelta(days=1)
        current_day = self.start
        while current_day != self.end:
            current_day_str = current_day.strftime("%Y-%m-%d")
            if np.is_busday(current_day_str):
                returned_bars = self.cache.get_bars(current_day_str)
                if returned_bars == None:
                    start_day = current_day + datetime.timedelta(hours=9, minutes=30)
                    end_day = current_day + datetime.timedelta(hours=16)
                    start_day_str = start_day.strftime("%Y-%m-%dT%H:%M:%S" + self.__daylight_savings_offset(current_day))
                    end_day_str = end_day.strftime("%Y-%m-%dT%H:%M:%S" + self.__daylight_savings_offset(current_day))
                    barset = self.__process_bars(self.api.get_barset(self.ticker, 'minute', start=start_day_str, end=end_day_str)[self.ticker])
                    if len(barset) == 0:
                        time.sleep(0.3)
                        current_day += delta
                        continue
                    self.data[current_day] = barset
                    self.date.append(current_day)
                    self.cache.add_bars(current_day_str, barset)
                    time.sleep(0.3)
                else:
                    self.data[current_day] = returned_bars
                    self.date.append(current_day)

            current_day += delta

    def __daylight_savings_offset(self, date):
        tz = pytz.timezone('America/New_York')
        offset_seconds = tz.utcoffset(date).seconds
        offset_hours = offset_seconds / 3600.0
        offset_hours -= 24
        if (offset_hours == -4.00):
            return "-04:00"
        else:
            return "-05:00"
    
    def __sequential(self, has_label=True):
        date = self.date[self.cursor]
        self.cursor += 1
        
        data = self.data[date]
        if has_label and self.labeler:
            label = self.label[date]
            return date, data, label
        else:
            return date, data
    
    def __process_bars(self, bars):
        return [{ 'c': b.c, 'h': b.h, 'l': b.l, 'o': b.o, 't': b.t, 'v': b.v } for b in bars]

    def get_random_params(self):
        date_keys = list(self.data.keys())
        date_key = random.choice(date_keys)
        seq_size = random.randrange(self.min_sequence, self.max_sequence)
        arr_size = len(self.data[date_key])
        arr_size -= (2 * self.num_bins + seq_size)
        idx = random.randrange(0, arr_size)
        start_idx = idx + self.num_bins
        end_idx = start_idx + seq_size
        return date_key, start_idx, end_idx
        
    def __reset(self):
        self.cursor = 0
        
    def __cursor_next(self):
        return self.cursor < len(self.date) 
    
    def has_next(self):
        return True

    def __get_label(self, date, start, end):
        label = self.label[date]
        labels = [np.array(l) for l in label]
        start_idx = start + StockDataLoader.futura
        end_idx = end + StockDataLoader.futura
        filtered_label = labels[start_idx:end_idx]
        dimension_length = self.max_sequence - len(filtered_label)
        diff_length = dimension_length * len(filtered_label[0])
        padding_zeros = np.zeros(diff_length)
        padding_zeros = padding_zeros.reshape((dimension_length, len(filtered_label[0])))
        return np.concatenate((filtered_label, padding_zeros))

    def __get_weights(self, length):
        weights = []
        for idx in range(0, length):
            weights.append(1.0)

        for idx in range(length, self.max_sequence):
            weights.append(0.0)

        return np.array(weights)

    def __norm_array(self, arr):
        corr = np.max(arr) - np.min(arr)
        nor = arr - np.min(arr)
        return nor / corr

    def __get_data(self, date, start, end):
        datum = self.data[date]
        flattened_datum = np.array([np.array([d['o'], d['c'], d['h'], d['l'], d['v']]) for d in datum])
        tot_arr = []
        for idx in range(start, end):
            sidx = idx - self.rolling_window
            sub_selection = flattened_datum[sidx:idx]
            o = [s[0] for s in sub_selection]
            c = [s[1] for s in sub_selection]
            h = [s[2] for s in sub_selection]
            l = [s[3] for s in sub_selection]
            v = [s[4] for s in sub_selection]
            o = self.__norm_array(o)
            c = self.__norm_array(c)
            h = self.__norm_array(h)
            l = self.__norm_array(l)
            v = self.__norm_array(v)
            zpt_arr = np.array([o, c, h, l, v])
            norm_arr = zpt_arr.transpose(1, 0).flatten()
            tot_arr.append(norm_arr)
        stream_datum  = np.array(tot_arr)
        dimension_length = self.max_sequence - len(stream_datum)
        diff_length = dimension_length * len(stream_datum[0])
        padding_zeros = np.zeros(diff_length)
        padding_zeros = padding_zeros.reshape((dimension_length, len(stream_datum[0])))
        return np.concatenate((stream_datum, padding_zeros))

    def next(self):
        weight_batch = []
        datum_batch = []
        label_batch = []
        for i in range(0, self.batch_size):
            date, start, end = self.get_random_params()
            length = end - start
            weights = self.__get_weights(length)
            datum = self.__get_data(date, start, end)
            label = self.__get_label(date, start, end)
            weight_batch.append(weights)
            datum_batch.append(datum)
            label_batch.append(label)
        return np.array(datum_batch), np.array(label_batch), np.array(weight_batch)