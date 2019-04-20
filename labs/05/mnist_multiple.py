#!/usr/bin/env python3
# 55d46f86-b962-11e7-a937-00505601122b
# 4fc059fa-abd2-11e7-a937-00505601122b
# be28f437-a9b0-11e7-a937-00505601122b
import numpy as np
import tensorflow as tf

from mnist import MNIST

# The neural network model
class Network:
    def __init__(self, args):
        # TODO: Add a `self.model` which has two inputs, both images of size [MNIST.H, MNIST.W, MNIST.C].
        i1 = inp1 = tf.keras.layers.Input(shape = (MNIST.H, MNIST.W, MNIST.C))
        i2 = inp2 = tf.keras.layers.Input(shape = (MNIST.H, MNIST.W, MNIST.C))
        
        # It then passes each input image through the same network (with shared weights), performing
        # - convolution with 10 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation
        l = tf.keras.layers.Conv2D(10, (3,3), padding='valid', strides=(2,2), activation='relu')
        i1 = l(i1)
        i2 = l(i2)
        # - convolution with 20 filters, 3x3 kernel size, stride 2, "valid" padding, ReLU activation
        l = tf.keras.layers.Conv2D(20, (3,3), padding='valid', strides=(2,2), activation='relu')
        i1 = l(i1)
        i2 = l(i2)
        # - flattening layer
        l = tf.keras.layers.Flatten()
        i1 = l(i1)
        i2 = l(i2)
        # - fully connected layer with 200 neurons and ReLU activation
        l = tf.keras.layers.Dense(200, activation='relu')
        i1 = l(i1)
        i2 = l(i2)
        # obtaining a 200-dimensional feature representation of each image.

        #
        # Then, it produces three outputs:
        # - classify the computed representation of the first image using a densely connected layer
        #   into 10 classes;
        # - classify the computed representation of the second image using the
        #   same connected layer (with shared weights) into 10 classes;
        img_class = tf.keras.layers.Dense(10, activation='softmax')
        o1 = img_class(i1)
        o2 = img_class(i2)
        # - concatenate the two image representations, process them using another fully connected
        #   layer with 200 neurons and ReLU, and finally compute one output with tf.nn.sigmoid
        #   activation (the goal is to predict if the first digit is larger than the second)
        #
        o3 = tf.keras.layers.concatenate([i1, i2])
        o3 = tf.keras.layers.Dense(200, activation='relu')(o3)
        o3 = tf.keras.layers.Dense(1, activation='sigmoid')(o3)
        # Train the outputs using SparseCategoricalCrossentropy for the first two inputs
        # and BinaryCrossentropy for the third one, utilizing Adam with default arguments.
        self.model = tf.keras.Model(inputs=[inp1, inp2], outputs=[o1, o2, o3])
        self.model.compile(optimizer='adam',
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),tf.keras.metrics.BinaryAccuracy(name="accuracy")],
              loss=[ 'sparse_categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy'])

    @staticmethod
    def _prepare_batches(batches_generator):
        batches = []
        for batch in batches_generator:
            batches.append(batch)
            if len(batches) >= 2:
                # TODO: yield the suitable modified inputs and targets using batches[0:2]
                model_inputs = [batches[0]['images'],batches[1]['images']]
                model_targets = [batches[0]['labels'],batches[1]['labels'],(batches[0]['labels'] > batches[1]['labels'])*1]
                yield (model_inputs, model_targets)
                batches.clear()

    def train(self, mnist, args):
        for epoch in range(args.epochs):
            # TODO: Train for one epoch using `model.train_on_batch` for each batch.
            for batch in self._prepare_batches(mnist.train.batches(args.batch_size)):
                self.model.train_on_batch(batch[0], batch[1])

            # Print development evaluation
            print("Dev {}: directly predicting: {:.4f}, comparing digits: {:.4f}".format(epoch + 1, *self.evaluate(mnist.dev, args)))

    def evaluate(self, dataset, args):
        # TODO: Evaluate the given dataset, returning two accuracies, the first being
        # the direct prediction of the model, and the second computed by comparing predicted
        # labels of the images.
        c = 0
        direct_accuracy = 0
        indirect_accuracy = 0
        for inputs, targets in self._prepare_batches(dataset.batches(args.batch_size)):
            p = self.model.predict_on_batch(inputs)
            for i in range(len(p[1])):
                direct_accuracy += round(p[2][i][0])==targets[2][i]
                indirect_accuracy += (np.argmax(p[0][i]) > np.argmax(p[1][i])) == targets[2][i]
                c += 1

        direct_accuracy /= c
        indirect_accuracy /= c

        return direct_accuracy, indirect_accuracy


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
    parser.add_argument("--recodex", default=True, action="store_true", help="Evaluation in ReCodEx.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds and number of threads
    np.random.seed(42)
    tf.random.set_seed(42)
    if args.recodex:
        tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.keras.initializers.glorot_uniform(seed=42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = MNIST()

    # Create the network and train
    network = Network(args)
    network.train(mnist, args)
    with open("mnist_multiple.out", "w") as out_file:
        direct, indirect = network.evaluate(mnist.test, args)
        print("{:.2f} {:.2f}".format(100 * direct, 100 * indirect), file=out_file)
