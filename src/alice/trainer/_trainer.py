"""
# -*- coding: utf-8 -*-
# Copyright © 2020 Abhishek Pratapa. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
"""

import os

class Trainer:
    model_path = '__checkpoints'
    def __init__(self, model, loader, max_iterations=10000):
        self.model = model
        self.loader = loader
        self.max_iterations = max_iterations
        self.index = 0
        self.loss = 0
        self.__create_dir()

    def __print_row(self):
        index_str ='{message: <{width}}'.format(message=str(self.index), width=11)
        loss_it = '%.5f'%(self.loss)
        loss_str = '{message: <{width}}'.format(message=loss_it, width=10)
        print("|" + index_str + "|" + loss_str + "|")

    def __print_header(self):
        print("+-----------+----------+")
        print("| iteration |   loss   |")
        print("+-----------+----------+")

    def __print_footer(self):
        print("+-----------+----------+")
        
    def __create_dir(self):
        try:  
            os.mkdir(Trainer.model_path + '/')  
        except:  
            pass

    def __save_checkpoint(self):
        file_name = Trainer.model_path + '/' + self.model.identifier() + '_checkpoint_' + str(self.index)
        self.model.save(file_name)

    def train(self):
        self.__print_header()
        while self.loader.has_next() and self.index < self.max_iterations:
            data, label, weights = self.loader.next()
            self.loss = self.model.train(data, label, weights)
            self.index += 1
            if self.index % 1000 == 0:
                self.__save_checkpoint()
            self.__print_row()
        self.__save_checkpoint()
        self.__print_footer()
