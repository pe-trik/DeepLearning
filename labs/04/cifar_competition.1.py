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
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, concatenate, Input, add
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import LearningRateScheduler

from cifar10 import CIFAR10

# The neural network model
class Network(tf.keras.Model):

    def block(self, input, size, dropout, regularization):

        x = Conv2D(size, (3,3), padding='same', activation='relu', kernel_regularizer=regularization, bias_regularizer=regularization)(input)

        x = BatchNormalization()(x)
        x = Conv2D(size, (3,3), padding='same', activation='relu', kernel_regularizer=regularization, bias_regularizer=regularization)(x)

        x = concatenate([x,input])
        x = BatchNormalization()(x)

        x = MaxPooling2D(pool_size=(2,2))(x)
        x = Dropout(dropout)(x)

        return x

    def __init__(self, args):
        # TODO: Define a suitable model, by calling `super().__init__`
        # with appropriate inputs and outputs.
        #
        # Alternatively, if you prefer to use a `tf.keras.Sequential`,
        # replace the `Network` parent, call `super().__init__` at the beginning
        # of this constructor and add layers using `self.add`.

        # TODO: After creating the model, call `self.compile` with appropriate arguments.
        
        input_img = Input(shape = (32, 32, 3))
        
        regularization = tf.keras.regularizers.L1L2(l2=1e-5)

        t3 = self.block(input_img, 128, 0.2, regularization)
        t3 = self.block(t3, 256, 0.3, regularization)
        t3 = self.block(t3, 512, 0.4, regularization)

        net = Conv2D(1024, (3,3), padding='same', activation='relu', kernel_regularizer=regularization, bias_regularizer=regularization)(t3)
        net = BatchNormalization()(net)
        net = AveragePooling2D(pool_size=(2,2))(net)
        net = Flatten()(net)
        net = Dropout(0.2)(net)

        net = Dense(10, activation = 'softmax', kernel_regularizer=regularization, bias_regularizer=regularization)(net)

        super().__init__(inputs=input_img,outputs=net)
        
        schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=0.001,
            decay_steps=args.epochs*45000/500,
            end_learning_rate=0.0001
        )
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=schedule),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, cifar, args):
        '''t = np.append(cifar.train.data["images"], np.flip(cifar.train.data["images"], 1), axis=0)
        d = np.append(cifar.dev.data["images"], np.flip(cifar.dev.data["images"], 1), axis=0)
        tr = tf.keras.utils.to_categorical(np.append(cifar.train.data["labels"], cifar.train.data["labels"], axis=0))
        de = tf.keras.utils.to_categorical(np.append(cifar.dev.data["labels"], cifar.dev.data["labels"], axis=0))'''
        t = cifar.train.data["images"]
        d = cifar.dev.data["images"]
        tr = tf.keras.utils.to_categorical(cifar.train.data["labels"])
        de = tf.keras.utils.to_categorical(cifar.dev.data["labels"])
        self.fit(
            t, tr,
            batch_size=args.batch_size, epochs=30,
            validation_data=(d, de),
            callbacks=[self.tb_callback],
        )


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
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
