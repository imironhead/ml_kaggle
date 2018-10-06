"""
"""
import skimage.io
import tensorflow as tf


def build_decoder():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    def decode_image_without_label(serialized_example):
        """
        """
        features = tf.parse_single_example(serialized_example, features={
            'image': tf.FixedLenFeature([], tf.string),
        })

        image = tf.decode_raw(features['image'], tf.uint8)

        image = tf.reshape(image, [FLAGS.image_size, FLAGS.image_size])

        return image

    def decode_image_with_label(serialized_example):
        """
        """
        features = tf.parse_single_example(serialized_example, features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

        label = tf.cast(features['label'], tf.int32)

        image = tf.decode_raw(features['image'], tf.uint8)

        image = tf.reshape(image, [FLAGS.image_size, FLAGS.image_size])

        return image, label

    if FLAGS.labeled:
        return decode_image_with_label
    else:
        return decode_image_without_label


def build_record_iterator(record_path):
    """
    read TFRecord batch.
    """
    # NOTE: read tfrecord
    data = tf.data.TFRecordDataset(record_path)

    data = data.map(build_decoder())

    # NOTE: create the final iterator
    iterator = data.make_initializable_iterator()

    return iterator


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    iterator = build_record_iterator(FLAGS.source_path)

    next_sample = iterator.get_next()

    with tf.Session() as session:
        session.run(iterator.initializer)

        sample = session.run(next_sample)

    if FLAGS.labeled:
        print('label: {}'.format(sample[1]))

    if FLAGS.labeled:
        skimage.io.imsave(FLAGS.result_path, sample[0])
    else:
        skimage.io.imsave(FLAGS.result_path, sample)


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('source_path', None, '')
    tf.app.flags.DEFINE_string('result_path', None, '')

    tf.app.flags.DEFINE_boolean('labeled', True, '')

    tf.app.flags.DEFINE_integer('image_size', 28, '')

    tf.app.run()

