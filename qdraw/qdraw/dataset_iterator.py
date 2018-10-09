"""
"""
import tensorflow as tf


def build_decoder(image_size):
    """
    """
    def decode_image_with_label(serialized_example):
        """
        """
        features = tf.parse_single_example(serialized_example, features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

        image = tf.decode_raw(features['image'], tf.uint8)

        image = tf.cast(image, tf.float32)

        image = (image - 127.5) / 127.5

        image = tf.reshape(image, [image_size, image_size, 1])

        label = tf.cast(features['label'], tf.int32)

        return image, label

    return decode_image_with_label


def build_iterator(training, record_paths, batch_size, image_size):
    """
    read TFRecord batch.
    """
    # NOTE: read tfrecord, can we load faster with large buffer_size?
    data = tf.data.TFRecordDataset(record_paths, buffer_size=2**20)

    if training:
        data = data.repeat()

    data = data.map(build_decoder(image_size))

    if training:
        data = data.shuffle(buffer_size=10000)

    data = data.batch(batch_size=batch_size)

    # NOTE: create the final iterator
    iterator = data.make_initializable_iterator()

    return iterator

