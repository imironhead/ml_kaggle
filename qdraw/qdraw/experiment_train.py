"""
"""
import os
import time

import numpy as np
import tensorflow as tf

import qdraw.dataset_iterator as dataset_iterator


def build_dataset():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    # NOTE: for changing dataset during training
    train_record_paths = tf.placeholder(tf.string, shape=[None])

    train_iterator = dataset_iterator.build_iterator(
        True, train_record_paths, FLAGS.batch_size, FLAGS.image_size)

    # NOTE: iterator for validation dataset
    valid_record_paths = [FLAGS.valid_tfr_path]

    valid_iterator = dataset_iterator.build_iterator(
        False, valid_record_paths, FLAGS.batch_size, FLAGS.image_size)

    # NOTE: a string handle as training/validation set switch
    dataset_handle = tf.placeholder(tf.string, shape=[])

    iterator = tf.data.Iterator.from_string_handle(
        dataset_handle,
        train_iterator.output_types,
        train_iterator.output_shapes)

    # NOTE: create an op to iterate the datasets
    next_image_batch, next_label_batch = iterator.get_next()

    return {
        'images': next_image_batch,
        'labels': next_label_batch,
        'train_iterator': train_iterator,
        'valid_iterator': valid_iterator,
        'dataset_handle': dataset_handle,
        'train_record_paths': train_record_paths,
    }


def build_model(dataset):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    if FLAGS.model == 'capsule':
        import qdraw.model_capsule as chosen_model
    elif FLAGS.model == 'cnn':
        import qdraw.model_cnn as chosen_model
    elif FLAGS.model == 'resnet':
        import qdraw.model_resnet as chosen_model

    return chosen_model.build_model(dataset['images'], dataset['labels'])


def build_summaries(model):
    """
    """
    return {
        'loss': tf.summary.scalar('loss', model['loss']),
    }


def refresh_train_dataset(session, dataset):
    """
    """


def train(session, step, model, dataset, dataset_handle, summaries, reporter):
    """
    """
    feeds = {
        dataset['dataset_handle']: dataset_handle,
        model['learning_rate']: 0.0001,
    }

    fetch = {
        'optimizer': model['optimizer'],
        'loss': model['loss'],
        'step': model['step'],
        'summary': summaries['loss'],
    }

    fetched = session.run(fetch, feed_dict=feeds)

    reporter.add_summary(fetched['summary'], fetched['step'])

    if fetched['step'] % 1000 == 0:
        print('loss[{}]: {}'.format(fetched['step'], fetched['loss']))

    return fetched['step']


def valid(session, step, model, dataset, dataset_handle, summaries, reporter):
    """
    """
    if step % 10000 != 0:
        return step

    session.run(dataset['valid_iterator'].initializer)

    num_images = 0
    ap = 0.0

    valid_begin = time.time()

    while True:
        try:
            feeds = {
                dataset['dataset_handle']: dataset_handle,
            }

            fetch = {
                'labels': model['labels'],
                'logits': model['logits'],
            }

            fetched = session.run(fetch, feed_dict=feeds)

            labels = fetched['labels']

            logits = np.argsort(fetched['logits'], axis=1)

            logits = logits[:, -1:-4:-1]

            num_images += logits.shape[0]

            for i in range(3):
                ap += np.sum(logits[:, i] == labels) / float(i + 1)
        except tf.errors.OutOfRangeError:
            break

    map_at_3 = ap / float(num_images)

    summaries = [tf.Summary.Value(tag='map', simple_value=map_at_3)]

    summaries = tf.Summary(value=summaries)

    reporter.add_summary(summaries, step)

    valid_elapsed = time.time() - valid_begin

    print('validate [{}]: {}'.format(valid_elapsed, map_at_3))

    return step


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    dataset = build_dataset()

    model = build_model(dataset)

    summaries = build_summaries(model)

    source_ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_path)
    target_ckpt_path = os.path.join(FLAGS.ckpt_path, 'model.ckpt')

    reporter = tf.summary.FileWriter(FLAGS.logs_path)

    with tf.Session() as session:
        if source_ckpt_path is None:
            session.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(session, source_ckpt_path)

        step = session.run(model['step'])

        # NOTE: exclude log which does not happend yet :)
        reporter.add_session_log(
            tf.SessionLog(status=tf.SessionLog.START), global_step=step)

        # NOTE: initialize dataset iterator
        train_record_paths = tf.gfile.ListDirectory(FLAGS.train_dir_path)
        train_record_paths = \
            [os.path.join(FLAGS.train_dir_path, n) for n in train_record_paths]

        session.run(
            dataset['train_iterator'].initializer,
            feed_dict={dataset['train_record_paths']: train_record_paths})
        session.run(dataset['valid_iterator'].initializer)

        # NOTE: generate handles for switching dataset
        train_handle = session.run(dataset['train_iterator'].string_handle())
        valid_handle = session.run(dataset['valid_iterator'].string_handle())

        while True:
            refresh_train_dataset(session, dataset)

            step = train(
                session,
                step,
                model,
                dataset,
                train_handle,
                summaries,
                reporter)

            step = valid(
                session,
                step,
                model,
                dataset,
                valid_handle,
                summaries,
                reporter)

            if step >= FLAGS.stop_at_step:
                break

        tf.train.Saver().save(
            session, target_ckpt_path, write_meta_graph=False)

    # NOTE: save meta
    tf.reset_default_graph()

    dataset['images'] = tf.placeholder(
        tf.float32,
        shape=[None, FLAGS.image_size, FLAGS.image_size, 1],
        name='images')
    dataset['labels'] = None

    model = build_model(dataset)

    tf.train.Saver().export_meta_graph(target_ckpt_path + '.meta')


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('train_dir_path', None, '')
    tf.app.flags.DEFINE_string('valid_tfr_path', None, '')
    tf.app.flags.DEFINE_string('ckpt_path', None, '')
    tf.app.flags.DEFINE_string('logs_path', None, '')
    tf.app.flags.DEFINE_string('model', None, '')
    tf.app.flags.DEFINE_string('learning_rate_policy', None, '')
    tf.app.flags.DEFINE_string('dataset_rotate_policy', None, '')

    tf.app.flags.DEFINE_integer('image_size', 28, '')
    tf.app.flags.DEFINE_integer('batch_size', 100, '')

    tf.app.flags.DEFINE_integer('stop_at_step', 1000, '')

    tf.app.run()

