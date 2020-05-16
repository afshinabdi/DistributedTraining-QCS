"""
   CIFAR-10 data set.
   See http://www.cs.toronto.edu/~kriz/cifar.html.
"""

import os
import tensorflow as tf

# dimensions of original MNIST images
HEIGHT = 28
WIDTH = 28


class MNISTDataSet(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self._training_filename = os.path.join(
            self.data_dir, 'mnist_train.tfrecord')
        self._test_filename = os.path.join(
            self.data_dir, 'mnist_test.tfrecord')

        self._samples_per_epoch = {'train': 60000, 'test': 10000}

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/format': tf.FixedLenFeature([], tf.string),
                'image/class/label': tf.FixedLenFeature([], tf.int64),
                'image/height': tf.FixedLenFeature([1], tf.int64),
                'image/width': tf.FixedLenFeature([1], tf.int64),
            })
        image = tf.image.decode_png(
            features['image/encoded'], channels=1, dtype=tf.uint8)
        image = tf.cast(image, tf.float32) / 255
        label = tf.cast(features['image/class/label'], tf.int32)

        return image, label

    def _get_training_dataset(self, batch_size, num_parallel_calls=8):
        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset(self._training_filename).repeat()

        # Parse records.
        dataset = dataset.map(
            self.parser, num_parallel_calls=num_parallel_calls)

        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
        dataset = dataset.shuffle(buffer_size=self._samples_per_epoch['train'])

        # Batch it up.
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)

        return dataset

    def _get_test_dataset(self, num_parallel_calls=8):
        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset(self._test_filename).repeat()

        # Parse records.
        dataset = dataset.map(
            self.parser, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size=self._samples_per_epoch['test'])

        return dataset

    def get_data(self, subset=['train', 'test'], batch_size=1, num_parallel_calls=8, device='/cpu:0'):
        with tf.device(device):
            databases = {}
            if 'train' in subset:
                databases['train'] = self._get_training_dataset(
                    batch_size, num_parallel_calls)

            if 'test' in subset:
                databases['test'] = self._get_test_dataset(num_parallel_calls)

            assert databases != {}, 'no valid subset of MNIST database is provided.'

            # get the types and shapes of the database outputs
            db_types = next(iter(databases.items()))[1].output_types
            db_shapes = next(iter(databases.items()))[1].output_shapes

            # define the iterator and generate different initializers
            iterator = tf.data.Iterator.from_structure(db_types, db_shapes)
            image_batch, label_batch = iterator.get_next()

            # define different initializers
            initializer_op = {}
            for k in databases.keys():
                initializer_op[k] = iterator.make_initializer(databases[k])

        return image_batch, label_batch, initializer_op

    def get_number_samples(self, subset='train'):
        assert subset in self._samples_per_epoch.keys(
        ), 'no valid subset of dataset is provided!'
        return self._samples_per_epoch[subset]
