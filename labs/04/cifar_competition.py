#!/usr/bin/env python3
# 55d46f86-b962-11e7-a937-00505601122b
# 4fc059fa-abd2-11e7-a937-00505601122b
# be28f437-a9b0-11e7-a937-00505601122b

import numpy as np
import tensorflow as tf
from keras import backend as K
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler

from cifar10 import CIFAR10

# The neural network model
class Network(tf.keras.Sequential):
    def __init__(self, args):
        # TODO: Define a suitable model, by calling `super().__init__`
        # with appropriate inputs and outputs.
        #
        # Alternatively, if you prefer to use a `tf.keras.Sequential`,
        # replace the `Network` parent, call `super().__init__` at the beginning
        # of this constructor and add layers using `self.add`.

        # TODO: After creating the model, call `self.compile` with appropriate arguments.
        super(Network,self).__init__()
        num_classes = 10
        
        weight_decay = 1e-4

        self.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(32, 32, 3)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())
        self.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())
        self.add(MaxPooling2D(pool_size=(2,2)))
        self.add(Dropout(0.2))
        
        self.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())
        self.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())
        self.add(MaxPooling2D(pool_size=(2,2)))
        self.add(Dropout(0.3))
        
        self.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())
        self.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        self.add(Activation('relu'))
        self.add(BatchNormalization())
        self.add(MaxPooling2D(pool_size=(2,2)))
        self.add(Dropout(0.4))
        
        self.add(Flatten())
        self.add(Dense(num_classes, activation='softmax'))

        schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=args.epochs*45000/500,
            decay_rate=0.0001/0.01
        )
        self.compile(
            optimizer=tf.keras.optimizers.Adam(clipnorm=1.0,clipvalue=0.5,learning_rate=schedule),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, cifar, args):
        self.fit(
            cifar.train.data["images"], cifar.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
            callbacks=[self.tb_callback],
        )


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=500, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=125, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load data
    cifar = CIFAR10()
    print(cifar.train.data["images"].shape[1:])

    # Create the network and train
    network = Network(args)
    network.train(cifar, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in network.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=out_file)
