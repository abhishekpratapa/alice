"""
# -*- coding: utf-8 -*-
# Copyright © 2020 Abhishek Pratapa. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause
"""

import tensorflow as tf
import numpy as np

from ._model import ModelTemplate

class TemporalModel(ModelTemplate):
    def __init__(self, batch_size, num_features, num_classes, prediction_window, learning_rate, conv_h, lstm_h, dense_h, seed):
        self.batch_size = batch_size
        self.num_features = num_features
        self.num_classes = num_classes

        self.conv_h = conv_h
        self.lstm_h = lstm_h
        self.dense_h = dense_h

        self.input = tf.keras.Input(shape=(prediction_window, num_features))
        self.output = self.__define_model(self.input, seed)
        self.model = tf.keras.Model(inputs=self.input, outputs=self.output)
        self.model.compile(
            loss=tf.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            sample_weight_mode="temporal"
        )

    def __define_model(self, input_node, seed):
        dense = tf.keras.layers.Conv1D(filters=self.conv_h, kernel_size=(1), padding='same', strides=1, use_bias=True, activation='relu')
        dropout = tf.keras.layers.Dropout(rate=0.2, seed=seed)
        lstm = tf.keras.layers.LSTM(units=self.lstm_h, return_sequences=True, use_bias=True)
        dense2 = tf.keras.layers.Dense(self.dense_h)
        batch_norm = tf.keras.layers.BatchNormalization()
        relu = tf.keras.layers.ReLU()
        dropout2 = tf.keras.layers.Dropout(rate=0.5, seed=seed)
        dense3 = tf.keras.layers.Dense(self.num_classes, use_bias=False)
        softmax = tf.keras.layers.Softmax()

        dense_output = dense(input_node)
        dropout_output = dropout(dense_output)
        lstm_output = lstm(dropout_output)
        dense2_output = dense2(lstm_output)
        batch_norm_output = batch_norm(dense2_output)
        relu_output = relu(batch_norm_output)
        dropout2_output = dropout2(relu_output)
        dense3_output = dense3(dropout2_output)

        return softmax(dense3_output)
    
    def train(self, inputs, labels, weights):
        loss = self.model.train_on_batch(
            x=inputs,
            y=labels,
            sample_weight=weights
        )
        return np.array(loss)

    def identifier(self):
        return 'temporal_model'
    
    def predict(self, inputs):
        loss = self.model.train_on_batch(
            x=inputs,
            y=tf.keras.utils.to_categorical(labels, num_classes=self.num_classes),
            sample_weight=np.reshape(weights, (self.batch_size, 1))
        )
        return np.array(loss)
    
    def save(self, path):
        self.model.save_weights(path)