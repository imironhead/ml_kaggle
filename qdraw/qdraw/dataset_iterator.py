"""
"""
import functools
import os

import tensorflow as tf


def decode(example, image_size):
    """
    """
    features = {
        'keyid': tf.FixedLenFeature([], tf.int64),
        'strokes': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
        'recognized': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
    }

    features = tf.parse_single_example(example, features=features)

    # NOTE: keyid, an int64
    keyid = features['keyid']

    # NOTE: strokes
    strokes = tf.decode_raw(features['strokes'], tf.uint8)
    strokes = tf.cast(strokes, tf.float32)
    strokes = tf.reshape(strokes, [-1, 3])
    strokes = strokes / 255.0

    # NOTE: strokes are padded to make batch for RNN
    #       use length to mask the output of rnn
    #       (set output[length:, :] to zeors)
    length = tf.shape(strokes)[0]

    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [image_size, image_size, 1])
    image = image / 255.0

    recognized = tf.cast(features['recognized'], tf.int32)
    label = tf.cast(features['label'], tf.int32)

    return keyid, image, strokes, length, recognized, label


def build_iterator(
        record_paths,
        batch_size,
        is_training,
        is_recognized_only,
        image_size):
    """
    read TFRecord batch.
    """
    fn_decode = functools.partial(decode, image_size=image_size)

    fn_reader = functools.partial(
        tf.data.TFRecordDataset,
        compression_type='GZIP',
        num_parallel_reads=32 if is_training else 1)

    data = tf.data.Dataset.from_tensor_slices(record_paths)

    if is_training:
        data = data.shuffle(buffer_size=10)
        data = data.repeat()
        data = data.interleave(fn_reader, cycle_length=10, block_length=1)
        data = data.shuffle(buffer_size=10000)
    else:
        data = fn_reader(data)

    data = data.map(fn_decode)

    if is_recognized_only:
        data = data.filter(lambda a, b, c, d, e, f: tf.equal(e, 1))

    data = data.prefetch(10000)

    # NOTE: shape of strokes is not fixed
    data = data.padded_batch(
        batch_size=batch_size, padded_shapes=data.output_shapes)

    return data.make_initializable_iterator()


def build_dataset(
        batch_size,
        image_size,
        valid_dir_path,
        test_dir_path,
        train_on_recognized=False):
    """
    """
    # NOTE: for changing dataset during training
    train_record_paths = tf.placeholder(tf.string, shape=[None])

    train_iterator = build_iterator(
        train_record_paths,
        batch_size=batch_size,
        is_training=True,
        is_recognized_only=train_on_recognized,
        image_size=image_size)

    # NOTE: iterator for validation dataset
    #       there is supposed only 1 tfrecord within this directory
    names = tf.gfile.ListDirectory(valid_dir_path)
    valid_record_paths = [os.path.join(valid_dir_path, n) for n in names]

    valid_iterator = build_iterator(
        valid_record_paths,
        batch_size=batch_size,
        is_training=False,
        is_recognized_only=False,
        image_size=image_size)

    # NOTE: iterator for testing dataset
    #       there is supposed only 1 tfrecord within this directory
    names = tf.gfile.ListDirectory(test_dir_path)
    test_record_paths = [os.path.join(test_dir_path, n) for n in names]

    test_iterator = build_iterator(
        test_record_paths,
        batch_size=batch_size,
        is_training=False,
        is_recognized_only=False,
        image_size=image_size)

    # NOTE: a string handle as training/validation set switch
    dataset_handle = tf.placeholder(tf.string, shape=[])

    iterator = tf.data.Iterator.from_string_handle(
        dataset_handle,
        train_iterator.output_types,
        train_iterator.output_shapes)

    return {
        'iterator': iterator,
        'train_iterator': train_iterator,
        'valid_iterator': valid_iterator,
        'test_iterator': test_iterator,
        'dataset_handle': dataset_handle,
        'train_record_paths': train_record_paths,
    }

