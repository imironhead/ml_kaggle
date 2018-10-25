"""
"""
import functools
import os

import tensorflow as tf


def decode(example, has_label, image_size):
    """
    """
    features = {
        'keyid': tf.FixedLenFeature([], tf.int64),
        'strokes': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
    }

    if has_label:
        features['recognized'] = tf.FixedLenFeature([], tf.int64)
        features['label'] = tf.FixedLenFeature([], tf.int64)

    features = tf.parse_single_example(example, features=features)

    keyid = features['keyid']

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

    if has_label:
        recognized = tf.cast(features['recognized'], tf.int32)
        label = tf.cast(features['label'], tf.int32)

        return keyid, image, strokes, length, recognized, label
    else:
        return keyid, image, strokes, length, -1, -1


def build_iterator(
        record_paths,
        batch_size,
        has_label,
        is_training,
        is_recognized_only,
        image_size):
    """
    read TFRecord batch.
    """
    fn_decode = functools.partial(
        decode,
        has_label=has_label,
        image_size=image_size)

    data = tf.data.Dataset.from_tensor_slices(record_paths)

    if is_training:
        data = data.shuffle(buffer_size=10)
        data = data.repeat()

        data = data.interleave(
            tf.data.TFRecordDataset, cycle_length=10, block_length=1)

        data = data.shuffle(buffer_size=10000)
    else:
        data = tf.data.TFRecordDataset(data)

    data = data.map(fn_decode)

    if is_recognized_only:
        data = data.filter(lambda a, b, c, d, e, f: tf.equal(e, 1))

    data = data.prefetch(10000)

    # NOTE: shape of strokes is not fixed
    data = data.padded_batch(
        batch_size=batch_size, padded_shapes=data.output_shapes)

    return data.make_initializable_iterator()

