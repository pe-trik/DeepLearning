#!/usr/bin/env python3

# 55d46f86-b962-11e7-a937-00505601122b
# 4fc059fa-abd2-11e7-a937-00505601122b
# be28f437-a9b0-11e7-a937-00505601122b

import numpy as np
import tensorflow as tf
import tensorflow_hub as tfhub

from caltech42 import Caltech42

# The neural network model
class Network:

    model = None
    netw_t = 2

    def __init__(self, args, l=True):
        mobilenet = tfhub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2", output_shape=[1280], trainable=True)
        self.netw_t = 1
        imgsize = 224
        oshape = 1280
        print(mobilenet.get_config())
        '''if l:
            self.model = tf.keras.Sequential([
                mobilenet,
                tf.keras.layers.Dropout(rate=0.5),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(oshape, activation='relu'),
                tf.keras.layers.Dropout(rate=0.5),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(42, activation='softmax')
            ])
        else:
            self.model = tf.keras.Sequential([
                mobilenet,
                tf.keras.layers.Dropout(rate=0.5),
                tf.keras.layers.Dense(oshape, activation='relu'),
                tf.keras.layers.Dropout(rate=0.5),
                tf.keras.layers.Dense(42, activation='softmax')
            ])'''
        inp = tf.keras.layers.Input(shape=(imgsize,imgsize,3))
        i = tf.keras.layers.BatchNormalization()(inp)
        t1 = mobilenet(inp)
        t1 = tf.keras.layers.Dropout(rate=0.5)(t1)

        t2 = tf.keras.layers.AveragePooling2D(pool_size=(4,4), padding='valid')(inp)
        t2 = tf.keras.layers.Flatten()(t2)
        t2 = tf.keras.layers.Dropout(rate=0.5)(t2)
        
        t = tf.keras.layers.concatenate([t1, t2])
        t = tf.keras.layers.BatchNormalization()(t)
        t = tf.keras.layers.Dense(oshape, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(t)
        t = tf.keras.layers.Dropout(rate=0.5)(t)
        t = tf.keras.layers.Dense(42, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001))(t)

        self.model = tf.keras.Model(inputs=inp,outputs=t)

        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9), 
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'])

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    @staticmethod
    def crop(image):
        decoded = tf.image.decode_image(image, channels=3,dtype=tf.float32)
        #standardized = tf.image.per_image_standardization(decoded)
        resized = tf.image.resize_with_pad(decoded,224,224)
        return resized.numpy()


    def train(self, caltech, args):
        t = np.array(caltech.train.data["images"])
        d = np.array(caltech.dev.data["images"])
        tr = np.array(tf.keras.utils.to_categorical(caltech.train.data["labels"]))
        de = np.array(tf.keras.utils.to_categorical(caltech.dev.data["labels"]))
        cb = tf.keras.callbacks.LambdaCallback(
            on_epoch_begin=lambda epoch, logs: print("Epoch {} begins".format(epoch)),
            on_epoch_end=lambda epoch, logs: self.model.save_weights('models5/model_{}_{}.md5'.format(self.netw_t,epoch))
        )
        
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=45,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True)
        datagen.fit(t)
        self.model.fit_generator(datagen.flow(t, tr, batch_size=args.batch_size),
                    steps_per_epoch=len(t) / args.batch_size, epochs=args.epochs,
            validation_data=(d, de),
            callbacks=[self.tb_callback,cb]
        )
        #for _ in range(args.epochs):
        #    for batch in caltech.train.batches(args.batch_size):        
        #        self.model.train_on_batch(batch['images'], tf.keras.utils.to_categorical(batch['labels'],num_classes = 42))

    def predict(self, caltech, args):
        return self.model.predict(np.array(caltech.data["images"]))



if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--logdir", default='logs', help="Batch size.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
    args = parser.parse_args()

    # Fix random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    #Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    ))
    print(args.logdir)


    # Load data
    caltech42 = Caltech42(image_processing=Network.crop)

    # Create the network and train
    network = Network(args, True)
    network2 = Network(args, False)
                    #2/21 3/17
    #network.model.load_weights('models3/model_1_17.md5')
    #network2.model.load_weights('models2/model_1_21.md5')
    network.train(caltech42, args)

    with open("caltech42_competition_test.txt", "w", encoding="utf-8") as out_file:
        g = 0
        g2 = 0
        g3 = 0
        g4 = 0
        g5 = 0
        for probs, probs2, gold in zip(network.predict(caltech42.test, args), network2.predict(caltech42.test, args), caltech42.test.data['labels']):
            if np.argmax(probs) == gold:
                g += 1
            if np.argmax(probs2) == gold:
                g2 += 1
            s = (probs + probs2) / 2
            if np.argmax(s) == gold:
                g3 += 1

            if np.max(probs) < np.max(probs2):
                if np.argmax(probs2) == gold:
                    g4 += 1
            else:
                if np.argmax(probs) == gold:
                    g4 += 1

            if np.max(probs) < np.max(s) and np.max(probs2) < np.max(s):
                print(np.argmax(s), file=out_file)
                if np.argmax(s) == gold:
                    g5 += 1
            else:
                if np.max(probs) < np.max(probs2):
                    print(np.argmax(probs2), file=out_file)
                    if np.argmax(probs2) == gold:
                        g5 += 1
                else:
                    print(np.argmax(probs), file=out_file)
                    if np.argmax(probs) == gold:
                        g5 += 1
        print(g / len(caltech42.dev.data['labels']))
        print(g2 / len(caltech42.dev.data['labels']))
        print(g3 / len(caltech42.dev.data['labels']))
        print(g4 / len(caltech42.dev.data['labels']))
        print(g5 / len(caltech42.dev.data['labels']))