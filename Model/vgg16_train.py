from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import pickle
import random
from tqdm import tqdm

import numpy as np
from six.moves import xrange # pylint: disable=redefined-builtin
import tensorflow as tf
from vgg16 import Vgg16

VGG_MEAN = [103.939, 116.779, 123.68]

FLAGS = None

def map_func(record):
    keys_to_features = {
        "image": tf.io.FixedLenFeature((224*224*3), tf.float32),
        "label": tf.io.FixedLenFeature((1), tf.int64)
    }
    parsed = tf.io.parse_single_example(record, keys_to_features)
    
    image = tf.reshape(parsed["image"], [224, 224, 3])
    label = tf.reshape(parsed["label"], [])

    return image, label

def main(FLAGS):
    strategy = tf.distribute.MirroredStrategy()
    print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    training_filenames = [os.path.join("tfdatasets/train", f) for f in os.listdir("tfdatasets/train")]
    testing_filenames = [os.path.join("tfdatasets/test", f) for f in os.listdir("tfdatasets/test")]
    print(len(training_filenames))
    print(len(testing_filenames))
    
    GLOBAL_BATCH_SIZE = FLAGS.batch_size * strategy.num_replicas_in_sync
    
    train_summary_writer = tf.summary.create_file_writer(os.path.join(FLAGS.model_dir, 'summaries/train'))
    test_summary_writer = tf.summary.create_file_writer(os.path.join(FLAGS.model_dir, 'summaries/test'))
    
    with strategy.scope():
        train_dataset = tf.data.TFRecordDataset(training_filenames)
        train_dataset = train_dataset.map(map_func)
        train_dataset = train_dataset.shuffle(buffer_size=1024)
        train_dataset = train_dataset.batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
        train_iterator = strategy.make_dataset_iterator(train_dataset)
        
        test_dataset = tf.data.TFRecordDataset(testing_filenames)
        test_dataset = test_dataset.map(map_func)
        test_dataset = test_dataset.shuffle(buffer_size=1024)
        test_dataset = test_dataset.batch(GLOBAL_BATCH_SIZE, drop_remainder=True)
        test_iterator = strategy.make_dataset_iterator(test_dataset)
        
        v = Vgg16()
        var_list = [val for k, val in v.var_dict.items()]
        
        optimizer = tf.optimizers.Adam(FLAGS.learn_rate)
        checkpoint = tf.train.Checkpoint(model=v)
        
        if FLAGS.start_checkpoint is not None:
            print("Restoring checkpoint")
            checkpoint.restore(FLAGS.start_checkpoint)
        
        train_loss = tf.metrics.Mean(name='train_loss')
        test_loss = tf.metrics.Mean(name='test_loss')

        train_accuracy = tf.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        test_accuracy = tf.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')
        
        def train_step(inputs):
            images, labels = inputs

            with tf.GradientTape() as tape:
                probs = v(images)
                loss = tf.reduce_sum(tf.losses.sparse_categorical_crossentropy(labels, probs)) * (1. / GLOBAL_BATCH_SIZE)

            gradients = tape.gradient(loss, var_list)
            optimizer.apply_gradients(zip(gradients, var_list))
            train_loss(loss)
            train_accuracy(labels, probs)

        # Test step
        def test_step(inputs):
            images, labels = inputs

            probs = v(images)
            loss = tf.reduce_sum(tf.losses.sparse_categorical_crossentropy(labels, probs)) * (1. / GLOBAL_BATCH_SIZE)
            predictions = tf.argmax(probs, axis=-1)
            test_loss(loss)
            test_accuracy(labels, probs)
        
        
        @tf.function
        def distributed_train():
            return strategy.experimental_run(train_step, train_iterator)

        @tf.function
        def distributed_test():
            return strategy.experimental_run(test_step, test_iterator)

        for epoch in range(FLAGS.epochs):
            it = tqdm()
            train_iterator.initialize()
            with train_summary_writer.as_default():
                while True:
                    try:
                        distributed_train()
                        it.update()
                        tf.summary.scalar('loss', train_loss.result(), step=optimizer.iterations)
                        tf.summary.scalar('accuracy', train_accuracy.result(), step=optimizer.iterations)
                    except tf.errors.OutOfRangeError:
                        it.close()
                        break
                    except Exception as e:
                        print(e)
                        continue

            it = tqdm()
            test_iterator.initialize()
            with test_summary_writer.as_default():
                while True:
                    try:
                        distributed_test()
                        it.update()
                        tf.summary.scalar('loss', test_loss.result(), step=optimizer.iterations)
                        tf.summary.scalar('accuracy', test_accuracy.result(), step=optimizer.iterations)
                    except tf.errors.OutOfRangeError:
                        it.close()
                        break
                    except Exception as e:
                        print(e)
                        continue

            # if epoch % 2 == 0:
            checkpoint.save(os.path.join(FLAGS.model_dir, "vgg16"))

            template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, "
                        "Test Accuracy: {}")
            print (template.format(epoch+1, train_loss.result(),
                                   train_accuracy.result()*100, test_loss.result(),
                                   test_accuracy.result()*100))

            train_loss.reset_states()
            test_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",
        type=str,
        default="m1/",
        help="Model save dir")
    parser.add_argument("--epochs",
        type=int,
        default=10,
        help="Model save dir")
    parser.add_argument("--start_checkpoint",
        type=str,
        default=None,
        help="Start checkpoint")
    parser.add_argument("--learn_rate",
        type=float,
        default=0.001,
        help="Learn rate")
    parser.add_argument("--batch_size",
        type=int,
        default=8,
        help="Batch size")
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)