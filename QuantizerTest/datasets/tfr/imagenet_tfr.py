"""
   Imagenet, ILSVRC2012 data set.
"""

import os
import tensorflow as tf

# db_setting = {
#     'data_dir': '',
#     'image_size': (227, 227),
#     'BGR': True,
#     'one_hot': False,
#     'resize_range': (256, 384),
#     'num_train_samples': 1281167,
#     'num_train_files': 1024,
#     'train_filenames': '{0:05d}',
#     'num_validation_samples': 50000,
#     'num_validation_files': 128,
#     'validation_filenames': '{0:05d}',
#     'augment_training': False,
#     'shuffle_buffer': 0.0001,
#     'num_classes': 1000,
#     'label_offset': 1,
# }


class ImagenetDataSet(object):
    def __init__(self, db_settings: dict):
        # read the settings of the database
        self.data_dir = db_settings.get('data_dir', '')
        self._one_hot = db_settings.get('one_hot', False)
        self._augment_training = db_settings.get('augment_training', False)
        self._image_size = db_settings.get('image_size', (227, 227))
        self._size_range = db_settings.get('resize_range', (256, 384))
        self._BGR = db_settings.get('BGR_format', True)
        
        # the labels in the tfr files is different than the ones some other models trained on. 
        # making the labels consistent with class_names in ilsvrc2012_classes
        self._lbl_offset = db_settings.get('label_offset', 1)

        # fill the list of training files names
        N = db_settings.get('num_train_files', 1024)
        filename = db_settings.get('train_filenames', '{0:05d}')
        self._train_filenames = [os.path.join(
            self.data_dir, filename.format(n, N)) for n in range(N)]

        # fill the list of training files names
        N = db_settings.get('num_validation_files', 128)
        filename = db_settings.get('validation_filenames', '{0:05d}')
        self._validation_filenames = [os.path.join(
            self.data_dir, filename.format(n, N)) for n in range(N)]

        self._samples_per_epoch = {
            'train': db_settings.get('num_train_samples', 1281167),
            'validation': db_settings.get('num_validation_samples', 50000)}

        self._num_classes = db_settings.get('num_classes', 1000)
        self._shuffle_buffer = db_settings.get('shuffle_buffer', 0.002)

    def parser(self, serialized_example):
        f = tf.parse_single_example(serialized_example, features={
            'image/filename': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/class/label': tf.FixedLenFeature([], tf.int64),
            'image/class/text': tf.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.FixedLenSequenceFeature([], tf.float32, True),
            'image/object/bbox/ymin': tf.FixedLenSequenceFeature([], tf.float32, True),
            'image/object/bbox/xmax': tf.FixedLenSequenceFeature([], tf.float32, True),
            'image/object/bbox/ymax': tf.FixedLenSequenceFeature([], tf.float32, True),
        })

        filename = f['image/filename']
        image = tf.image.decode_jpeg(f['image/encoded'], channels=3)
        label = f['image/class/label'] - self._lbl_offset
        label = tf.floormod(label, self._num_classes)
        if self._one_hot:
            label = tf.one_hot(label, self._num_classes)

        label_text = f['image/class/text']

        ymin = tf.reduce_min(f['image/object/bbox/ymin'])
        xmin = tf.reduce_min(f['image/object/bbox/xmin'])
        ymax = tf.reduce_max(f['image/object/bbox/ymax'])
        xmax = tf.reduce_max(f['image/object/bbox/xmax'])
        bounding_box = [ymin, xmin, ymax, xmax]

        return image, label, label_text, filename, bounding_box

    def _resize_image_keep_aspect(self, image, min_size):
        # Take width/height
        initial_width = tf.shape(image)[0]
        initial_height = tf.shape(image)[1]

        # Take the greater value, and use it for the ratio
        min_ = tf.minimum(initial_width, initial_height)
        ratio = tf.to_float(tf.truediv(min_, min_size))

        new_width = tf.to_int32(tf.to_float(initial_width) / ratio)
        new_height = tf.to_int32(tf.to_float(initial_height) / ratio)

        return tf.image.resize_images(image, size=(new_width, new_height),
                                      method=tf.image.ResizeMethod.BILINEAR)

    def _get_training_dataset(self, batch_size, num_parallel_calls=8):
        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset(
            self._train_filenames).repeat()

        # Parse records.
        dataset = dataset.map(
            self.parser, num_parallel_calls=num_parallel_calls)

        # augment dataset
        if self._augment_training:
            dataset = dataset.map(
                self._augment_training_image, num_parallel_calls=num_parallel_calls)
        else:
            dataset = dataset.map(self._prepare_validation_image,
                                  num_parallel_calls=num_parallel_calls)

        min_queue_examples = int(self._samples_per_epoch['train'] *
                                 self._shuffle_buffer)
        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
        dataset = dataset.shuffle(
            buffer_size=min_queue_examples + 3 * batch_size)

        # Batch it up.
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)

        return dataset

    def _get_validation_dataset(self, batch_size=None, num_parallel_calls=8):
        if batch_size is None:
            batch_size = self._samples_per_epoch['validation']

        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset(
            self._validation_filenames).repeat()

        # Parse records.
        dataset = dataset.map(
            self.parser, num_parallel_calls=num_parallel_calls)
        dataset = dataset.map(self._prepare_validation_image,
                              num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size)

        return dataset

    def _augment_training_image(self, image, *args):
        # 1- resize image to a random dimension
        min_size = tf.random_uniform([], minval=self._size_range[0],
                                     maxval=self._size_range[1]+1, dtype=tf.int32)
        image = self._resize_image_keep_aspect(image, min_size)

        # 2- crop the image to the desired output size
        image = tf.random_crop(
            image, [self._image_size[0], self._image_size[1], 3])
        image.set_shape([self._image_size[0], self._image_size[1], 3])

        # 3- randomly flip the image
        image = tf.image.random_flip_left_right(image)

        # 4- if necessary, change RGB to BGR format
        if self._BGR:
            image = tf.reverse(image, axis=[-1])

        # 5- subtract the mean
        imagenet_mean = tf.constant([123.68, 116.779, 103.939],
                                    dtype=tf.float32)
        image = tf.subtract(tf.to_float(image), imagenet_mean)

        return (image, ) + args

    def _prepare_validation_image(self, image, *args):
        # 1- resize image
        image = self._resize_image_keep_aspect(image, self._size_range[0])

        # 2- crop the center of image with the desired output size
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = (image_height - self._image_size[0]) // 2
        offset_width = (image_width - self._image_size[1]) // 2
        image = tf.image.crop_to_bounding_box(image, offset_height, offset_width,
                                              self._image_size[0], self._image_size[1])
        image.set_shape([self._image_size[0], self._image_size[1], 3])

        # 3- if necessary, change RGB to BGR format
        if self._BGR:
            image = tf.reverse(image, axis=[-1])

        # 4- subtract the mean
        imagenet_mean = tf.constant([123.68, 116.779, 103.939],
                                    dtype=tf.float32)
        image = tf.subtract(tf.to_float(image), imagenet_mean)

        return (image, ) + args

    def get_data(self, subset, batch_size, num_parallel_calls=8, device='/cpu:0'):
        with tf.device(device):
            databases = {}
            if 'train' in subset:
                databases['train'] = self._get_training_dataset(batch_size, num_parallel_calls)

            if 'validation' in subset:
                databases['validation'] = self._get_validation_dataset(batch_size, num_parallel_calls)

            assert databases != {}, 'no valid subset of database is provided.'

            # get the types and shapes of the database outputs
            tmp_db = next(iter(databases.items()))[1]
            db_types = tmp_db.output_types
            db_shapes = tmp_db.output_shapes
            db_classes = tmp_db.output_classes

            # define the iterator and generate different initializers
            iterator = tf.data.Iterator.from_structure(
                db_types, output_shapes=db_shapes, output_classes=db_classes)
            features = iterator.get_next()

            # define different initializers
            initializer_op = {}
            for k in databases.keys():
                initializer_op[k] = iterator.make_initializer(databases[k])

        return features, initializer_op

    def get_number_samples(self, subset='train'):
        assert subset in self._samples_per_epoch.keys(), 'no valid subset of dataset is provided!'
        return self._samples_per_epoch[subset]
