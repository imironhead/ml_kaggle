"""
"""
import numpy as np
import tensorflow as tf


def build_dataset():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    def decode_image(serialized_example):
        """
        """
        features = tf.parse_single_example(serialized_example, features={
            'image': tf.FixedLenFeature([], tf.string),
        })

        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.cast(image, tf.float32)
        image = (image - 127.5) / 127.5
        image = tf.reshape(image, [FLAGS.image_size, FLAGS.image_size, 1])

        return image

    data = tf.data.TFRecordDataset(FLAGS.source_path)

    data = data.map(decode_image)

    data = data.batch(batch_size=100)

    # NOTE: create the final iterator
    iterator = data.make_initializable_iterator()

    return {
        'iterator': iterator,
        'images': iterator.get_next(),
    }


def build_model(session):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    # NOTE: restore model weights
    saver = tf.train.import_meta_graph(FLAGS.ckpt_path + '.meta')

    saver.restore(session, FLAGS.ckpt_path)

    graph = tf.get_default_graph()

    return {
        'images': graph.get_tensor_by_name('images:0'),
        'logits': graph.get_tensor_by_name('logits:0'),
    }


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    all_logits = []

    dataset = build_dataset()

    with tf.Session() as session:
        model = build_model(session)

        session.run(dataset['iterator'].initializer)

        while True:
            try:
                images = session.run(dataset['images'])

                feeds = {
                    model['images']: images
                }

                logits = session.run(model['logits'], feed_dict=feeds)

                all_logits.append(logits)
            except tf.errors.OutOfRangeError:
                break

    all_logits = np.concatenate(all_logits, axis=0)

    np.savez(FLAGS.result_path, logits=all_logits)


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('ckpt_path', None, '')

    tf.app.flags.DEFINE_string('source_path', None, '')
    tf.app.flags.DEFINE_string('result_path', None, '')

    tf.app.flags.DEFINE_integer('image_size', 28, '')

    tf.app.run()

