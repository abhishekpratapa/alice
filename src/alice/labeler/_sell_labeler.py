"""
# -*- coding: utf-8 -*-
# Copyright © 2020 Abhishek Pratapa. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
"""

import numpy as np
from scipy import fftpack
import time

from ._labeler import LabelerTemplate

class SellLabeler(LabelerTemplate):
    def __init__(self, timestep, num_bins, step_size):
        self.timestep = timestep
        self.num_bins = num_bins
        self.step_size = step_size

    def __normalize_signal(self, signal):
        presignal = np.array(signal)
        return 2.*(presignal - np.min(presignal))/np.ptp(presignal)-1

    def __compute_derivative(self, tme, ticker):
        sec_time = [time.mktime(t.timetuple()) / 60.0 for t in tme]
        return np.diff(ticker)/np.diff(sec_time)
    
    def __unpack_fft(self, sig):
        sig_fft = fftpack.fft(sig)
        power = np.abs(sig_fft)**2
        sample_freq = fftpack.fftfreq(sig.size, d=self.timestep)
        return sig_fft, power, sample_freq

    def __compute_peak_frequency(self, sample_freq, power):
        pos_mask = np.where(sample_freq > 0)
        freqs = sample_freq[pos_mask]
        return freqs[power[pos_mask].argmax()]

    def __plot_harmonics(self, sig_fft, sample_freq, multiple, peak_freq):
        high_freq_fft = sig_fft.copy()
        high_freq_fft[np.abs(sample_freq) > peak_freq * multiple] = 0
        filtered_sig_img = fftpack.ifft(high_freq_fft)
        filtered_sig = [np.sign(np.real(f)) * np.abs(f) for f in filtered_sig_img]
        return filtered_sig

    def __cross_dict(self, timings_cross, sign_changes):
        cross = {}
        for t, s in zip(timings_cross, sign_changes):
            cross[t] = s
        return cross
    
    def __get_time_and_price(self, bars):
        time = [b['t'] for b in bars]
        price = [np.average([b['c'], b['o']]) for b in bars]
        return time, price
    
    def __filter_dict(self, dictObj, callback):
        newDict = dict()
        for (key, value) in dictObj.items():
            if callback((key, value)):
                newDict[key] = value
        return newDict
    
    def __weight_function(self, harmonic_values):
        harmonic_float = harmonic_values.astype(np.float32)
        harmonic_prelim_weights = np.reciprocal(harmonic_float)
        harmonic_squared = np.square(harmonic_prelim_weights)
        harmonic_prelim_sum = np.sum(harmonic_squared)
        return harmonic_prelim_weights / harmonic_prelim_sum
    
    def __buy_sell_pairs(self, bars, multiple):
        average_bar_time, average_bar_price = self.__get_time_and_price(bars)

        sig = self.__normalize_signal(average_bar_price)
        sig_fft, power, sample_freq = self.__unpack_fft(sig)
        peak_freq = self.__compute_peak_frequency(sample_freq, power)

        smoothed_plot = self.__plot_harmonics(sig_fft, sample_freq, multiple, peak_freq)
        harmonic_derivative = self.__compute_derivative(average_bar_time, smoothed_plot)
        harmonic_derivative_time = average_bar_time[:-1]

        zero_crossings = np.where(np.diff(np.signbit(harmonic_derivative)))[0]
        sign_changes = [-1 if harmonic_derivative[z] > harmonic_derivative[z + 1] else 1 for z in zero_crossings]
        timings_cross = [harmonic_derivative_time[t] for t in zero_crossings]

        return self.__cross_dict(timings_cross, sign_changes)

    def __process_frequency(self, bars):
        harmonic_values = np.arange(1.0, self.num_bins, self.step_size)
        harmonic_weights = self.__weight_function(harmonic_values)
        total_buy_dict = {}
        total_sell_dict = {}
        for h, w in zip(harmonic_values, harmonic_weights):
            bs_dict = self.__buy_sell_pairs(bars, multiple=h)
            sell_dict = self.__filter_dict(bs_dict, lambda elem : elem[1] == -1)
            buy_dict = self.__filter_dict(bs_dict, lambda elem : elem[1] == 1)
            for bk in buy_dict.keys():
                if bk in total_buy_dict:
                    total_buy_dict[bk] += w
                else:
                    total_buy_dict[bk] = w

            for sk in sell_dict.keys():
                if sk in total_sell_dict:
                    total_sell_dict[sk] += w
                else:
                    total_sell_dict[sk] = w

        return total_sell_dict, total_buy_dict
    
    def __get_date(self, bars):
        date_dict = dict()
        for b in bars: 
            date_dict[b['t']] = 0
        return date_dict
    
    def __norm_array(self, arr):
        corr = np.max(arr) - np.min(arr)
        nor = arr - np.min(arr)
        return nor / corr
    
    def identifier(self):
        return 'Sell_Probabilities'

    def process_data(self, bars):
        return_dict = self.__get_date(bars)
        sell_dict, buy_dict = self.__process_frequency(bars)
        
        for sd in sell_dict.keys():
            return_dict[sd] = sell_dict[sd]

        sell_arr = []
        for rd in return_dict.keys():
            sell_arr.append(return_dict[rd])
        sell_arr = self.__norm_array(sell_arr)

        sell_label = []
        for s in sell_arr:
            if s > 0.7:
                sell_label.append([0, 1])
            else:
                sell_label.append([1, 0])

        return np.array(sell_label)
