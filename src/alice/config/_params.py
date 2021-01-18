"""
# -*- coding: utf-8 -*-
# Copyright © 2020 Abhishek Pratapa. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import json

def load_config(path=None):
    data = dict()
    if path != None:
        with open(path) as json_file:
            data = json.load(json_file)
    config = Config()
    for k in data.keys():
        if k == 'batch_size':
            config.batch_size = data[k]
        if k == 'timestep':
            config.timestep = data[k]
        if k == 'multiple':
            config.multiple = data[k]
        if k == 'num_bins':
            config.num_bins = data[k]
        if k == 'rolling_window':
            config.rolling_window = data[k]
        if k == 'num_features':
            config.num_features = data[k]
        if k == 'step_size':
            config.step_size = data[k]
        if k == 'prediction_window':
            config.prediction_window = data[k]
        if k == 'seed':
            config.seed = data[k]
        if k == 'max_iterations':
            config.max_iterations = data[k]
        if k == 'learning_rate':
            config.learning_rate = data[k]
        if k == 'alpaca_key':
            config.alpaca_key = data[k]
        if k == 'alpaca_secrets':
            config.alpaca_secrets = data[k]
        if k == 'tickers':
            config.tickers = data[k]
    return config

class Config:
    def __init__(self):
        self.batch_size = 32
        self.timestep = 60
        self.multiple = 13
        self.num_bins = 64
        self.rolling_window = 12
        self.num_features = 5 * self.rolling_window
        self.step_size = 1.0
        self.prediction_window = 130
        self.seed = 1248
        self.max_iterations = 2000
        self.learning_rate = 1e-3
        self.model_config = {
            'conv_h': 64,
            'lstm_h': 200,
            'dense_h': 128,
            'num_classes': 2
        }
        self.alpaca_key = ''
        self.alpaca_secrets = ''
        self.tickers = []
        self.start_date = datetime.datetime(2021, 1, 13)
        self.end_date = datetime.datetime(2021, 1, 15)
        self.min_sequence = 5
        self.max_sequence = 130
        self.rolling_window = 12
