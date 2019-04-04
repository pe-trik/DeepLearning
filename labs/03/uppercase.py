#!/usr/bin/env python3
#
# All team solutions **must** list **all** members of the team.
# The members must be listed using their ReCodEx IDs anywhere
# in a comment block in the source file (on a line beginning with `#`).
#
# You can find out ReCodEx ID in the URL bar after navigating
# to your User profile page. The ID has the following format:
# 55d46f86-b962-11e7-a937-00505601122b
# 4fc059fa-abd2-11e7-a937-00505601122b
import argparse
import datetime
import os
import re
import random
import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=48, type=int, help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--batch_size", default=512, type=int, help="Batch size.")
parser.add_argument("--dropout", default=0.3, type=float, help="Dropout regularization.")
parser.add_argument("--l2", default=0.1, type=float, help="L2 regularization.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default="10,10,10", type=str, help="Hidden layer configuration.")
#parser.add_argument("--hidden_layer_neurons", default="100", type=str, help="Hidden layer configuration.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=10, type=int, help="Window size to use.")
args = parser.parse_args()
args.hidden_layers = [int(hidden_layer) for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

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
uppercase_data = UppercaseData(args.window, args.alphabet_size)

indices_with_ones = list(np.concatenate(np.where(uppercase_data.train.data["labels"] == 1)))
zeros = list(np.concatenate(np.where(uppercase_data.train.data["labels"] == 0)))
indices_with_zeros = list(zeros) #random.sample(zeros,len(indices_with_ones)*3)
print(len(indices_with_ones))
print(len(indices_with_zeros))
result_indices = indices_with_ones+indices_with_zeros#+indices_with_ones
print(len(result_indices))
train_data = dict()
train_data["labels"] = uppercase_data.train.data["labels"]#[result_indices]
train_data["windows"] = uppercase_data.train.data["windows"]#[result_indices]

if args.l2 != 0:
    regularization = tf.keras.regularizers.L1L2(l2=args.l2)
else:
    regularization = None

args.learning_rate = 0.1
    
steps =  (len(train_data["labels"])*args.epochs)/args.batch_size
exp_decay = (0.001/args.learning_rate)
learn_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = args.learning_rate,
    decay_steps=steps,
    decay_rate=exp_decay
)

# Create the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32))
model.add(tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(1050, 
                                    activation=tf.nn.relu
                                    ))

model.add(tf.keras.layers.Dropout(rate=args.dropout))

model.add(tf.keras.layers.Dense(1050, 
                                    activation=tf.nn.tanh
                                    ))

model.add(tf.keras.layers.Dropout(rate=args.dropout))

model.add(tf.keras.layers.Dense(42, 
                                    activation=tf.nn.relu
                                    ))
model.add(tf.keras.layers.Dropout(rate=args.dropout))


model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax)
          )

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1000000, update_freq=1000000, profile_batch=1)
tb_callback.on_train_end = lambda *_: None
model.fit(
    train_data["windows"], train_data["labels"],
    batch_size=args.batch_size, epochs=args.epochs,
    validation_data=(uppercase_data.dev.data["windows"], uppercase_data.dev.data["labels"]),
    callbacks=[tb_callback],
)

#test_logs = model.evaluate(
#    uppercase_data.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size,
#)
#tb_callback.on_epoch_end(1, dict(("val_test_" + metric, value) for metric, value in zip(model.metrics_names, test_logs)))

# TODO: Implement a suitable model, optionally including regularization, select
# good hyperparameters and train the model.
#
# The inputs are _windows_ of fixed size (`args.window` characters on left,
# the character in question, and `args.window` characters on right), where
# each character is representedy by a `tf.int32` index. To suitably represent
# the characters, you can:
# - Convert the character indices into _one-hot encoding_. There is no
#   explicit Keras layer, so you can
#   - use a Lambda layer which can encompass any function:
#       Sequential([
#         tf.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
#         tf.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
#   - or use Functional API and a code looking like
#       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
#       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
#   You can then flatten the one-hot encoded windows and follow with a dense layer.
# - Alternatively, you can use `tf.keras.layers.Embedding`, which is an efficient
#   implementation of one-hot encoding followed by a Dense layer, and flatten afterwards.
import io
with io.open("uppercase_test.txt", "w", encoding="utf-8") as out_file:
    results = np.argmax(model.predict(uppercase_data.test.data["windows"]), axis=1)
    for i in range(len(results)):
        ch = uppercase_data.test.text[i]
        if results[i]==1:
            ch = ch.upper()
        out_file.write(ch)
    # TODO: Generate correctly capitalized test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    #    # and write the result to `uppercase_test.txt` file.

    pass

