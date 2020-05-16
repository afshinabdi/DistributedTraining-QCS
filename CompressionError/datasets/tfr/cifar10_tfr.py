"""
   CIFAR-10 data set.
   See http://www.cs.toronto.edu/~kriz/cifar.html.
"""

import os
import tensorflow as tf

# dimensions of original CIFAR10 images
HEIGHT = 32
WIDTH = 32
DEPTH = 3


class Cifar10Dataset(object):
    def __init__(self, data_dir, image_size=(24, 24), augment_training=True):
        self.data_dir = data_dir
        self._augment_training = augment_training
        self._output_dim = image_size

        self._training_filename = os.path.join(self.data_dir, 'train.tfrecord')
        self._test_filename = os.path.join(self.data_dir, 'test.tfrecord')

        self._samples_per_epoch = {'train': 50000, 'test': 10000}

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
            }
        )

        image = tf.image.decode_png(features['image/encoded'], dtype=tf.uint8)
        image = tf.cast(image, tf.float32) / 255
        label = tf.cast(features['image/class/label'], tf.int32)

        return image, label

    def _get_training_dataset(self, batch_size, num_parallel_calls=8):
        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset(self._training_filename).repeat()

        # Parse records.
        dataset = dataset.map(self.parser, num_parallel_calls=num_parallel_calls)

        # augment dataset
        if self._augment_training:
            dataset = dataset.map(self._augment_training_image, num_parallel_calls=num_parallel_calls)
        else:
            dataset = dataset.map(self._prepare_test_image, num_parallel_calls=num_parallel_calls)

        min_queue_examples = int(self._samples_per_epoch['train'] * 0.4)
        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
        dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

        # Batch it up.
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)

        return dataset

    def _get_test_dataset(self, num_parallel_calls=8):
        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset(self._test_filename).repeat()

        # Parse records.
        dataset = dataset.map(self.parser, num_parallel_calls=num_parallel_calls)
        dataset = dataset.map(self._prepare_test_image, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size=self._samples_per_epoch['test'])

        return dataset

    def _augment_training_image(self, image, label):
        # Randomly crop a [height, width] section of the image.
        if (self._output_dim[0] >= HEIGHT) or (self._output_dim[1] >= WIDTH):
            image = tf.image.resize_with_crop_or_pad(image, self._output_dim[0] + 8, self._output_dim[1] + 8)

        image = tf.random_crop(image, [self._output_dim[0], self._output_dim[1], 3])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels.
        image = tf.image.per_image_standardization(image)

        # Set the shapes of tensors.
        image.set_shape([self._output_dim[0], self._output_dim[1], 3])
        # label.set_shape([1])

        return image, label

    def _prepare_test_image(self, image, label):
        # Crop the central [height, width] section of the image.
        image = tf.image.resize_image_with_crop_or_pad(image, self._output_dim[0], self._output_dim[1])

        # Subtract off the mean and divide by the variance of the pixels.
        image = tf.image.per_image_standardization(image)

        # Set the shapes of tensors.
        image.set_shape([self._output_dim[0], self._output_dim[1], 3])
        # label.set_shape([1])

        return image, label

    def create_dataset(self, subset=['train', 'test'], batch_size=1, num_parallel_calls=8):
		databases = {}
		if 'train' in subset:
			databases['train'] = self._get_training_dataset(batch_size, num_parallel_calls)

		if 'test' in subset:
			databases['test'] = self._get_test_dataset(num_parallel_calls)

		assert databases != {}, 'no valid subset of CIFAR10 database is provided.'

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
        assert subset in self._samples_per_epoch.keys(), 'no valid subset of dataset is provided!'
        return self._samples_per_epoch[subset]