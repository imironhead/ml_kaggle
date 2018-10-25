"""
"""
import csv
import gzip
import os

import numpy as np
import tensorflow as tf

import qdraw.dataset as dataset
import qdraw.dataset_iterator as dataset_iterator


def build_dataset():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    # NOTE: for changing dataset during training
    train_record_paths = tf.placeholder(tf.string, shape=[None])

    train_iterator = dataset_iterator.build_iterator(
        train_record_paths,
        batch_size=FLAGS.batch_size,
        has_label=True,
        is_training=True,
        is_recognized_only=FLAGS.train_on_recognized,
        image_size=FLAGS.image_size)

    # NOTE: iterator for validation dataset
    valid_record_paths = [FLAGS.valid_tfr_path]

    valid_iterator = dataset_iterator.build_iterator(
        valid_record_paths,
        batch_size=FLAGS.batch_size,
        has_label=True,
        is_training=False,
        is_recognized_only=False,
        image_size=FLAGS.image_size)

    # NOTE: iterator for testing dataset
    test_record_paths = [FLAGS.test_tfr_path]

    test_iterator = dataset_iterator.build_iterator(
        test_record_paths,
        batch_size=FLAGS.batch_size,
        has_label=False,
        is_training=False,
        is_recognized_only=False,
        image_size=FLAGS.image_size)

    # NOTE: a string handle as training/validation set switch
    dataset_handle = tf.placeholder(tf.string, shape=[])

    iterator = tf.data.Iterator.from_string_handle(
        dataset_handle,
        train_iterator.output_types,
        train_iterator.output_shapes)

    # NOTE: create an op to iterate the datasets
    keyids, images, strokes, lengths, recognized, labels = iterator.get_next()

    return {
        'keyids': keyids,
        'images': images,
        'strokes': strokes,
        'lengths': lengths,
        'labels': labels,
        'train_iterator': train_iterator,
        'valid_iterator': valid_iterator,
        'test_iterator': test_iterator,
        'dataset_handle': dataset_handle,
        'train_record_paths': train_record_paths,
    }


def build_model(data):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    if FLAGS.model == 'cnn':
        import qdraw.model_cnn as chosen_model
    elif FLAGS.model == 'rnn':
        import qdraw.model_rnn as chosen_model
    elif FLAGS.model == 'resnet':
        import qdraw.model_resnet as chosen_model
    elif FLAGS.model == 'mobilenets':
        import qdraw.model_mobilenets as chosen_model
    elif FLAGS.model == 'mobilenets_v2':
        import qdraw.model_mobilenets_v2 as chosen_model

    return chosen_model.build_model(
        data['images'],
        data['strokes'],
        data['lengths'],
        data['labels'])


def build_summaries(model):
    """
    """
    return {
        'loss': tf.summary.scalar('loss', model['loss']),
    }


def train(session, step, model, data, dataset_handle, summaries, reporter):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    feeds = {
        data['dataset_handle']: dataset_handle,
        model['learning_rate']: FLAGS.initial_learning_rate,
    }

    fetch = {
        'optimizer': model['optimizer'],
        'loss': model['loss'],
        'step': model['step'],
        'summary': summaries['loss'],
    }

    if 'training' in model:
        feeds[model['training']] = True

    fetched = session.run(fetch, feed_dict=feeds)

    reporter.add_summary(fetched['summary'], fetched['step'])

    if fetched['step'] % 1000 == 0:
        print('loss[{}]: {}'.format(fetched['step'], fetched['loss']))

    return fetched['step']


def valid(session, step, model, data, dataset_handle, summaries, reporter):
    """
    """
    if step % 10000 != 0:
        return step

    session.run(data['valid_iterator'].initializer)

    num_images = 0
    aps = [0.0, 0.0, 0.0]

    while True:
        try:
            feeds = {
                data['dataset_handle']: dataset_handle,
            }

            fetch = {
                'labels': model['labels'],
                'logits': model['logits'],
            }

            if 'training' in model:
                feeds[model['training']] = False

            fetched = session.run(fetch, feed_dict=feeds)

            labels = fetched['labels']

            logits = np.argsort(fetched['logits'], axis=1)

            logits = logits[:, -1:-4:-1]

            num_images += logits.shape[0]

            for i in range(3):
                aps[i] += np.sum(logits[:, i] == labels) / float(i + 1)
        except tf.errors.OutOfRangeError:
            break

    map_at_3 = sum(aps) / float(num_images)

    summaries = [tf.Summary.Value(tag='map', simple_value=map_at_3)]

    summaries = tf.Summary(value=summaries)

    reporter.add_summary(summaries, step)

    print('validation at step: {}'.format(step))
    print('map_1: {}'.format(aps[0] / float(num_images)))
    print('map_2: {}'.format(aps[1] / float(num_images)))
    print('map_3: {}'.format(aps[2] / float(num_images)))
    print('map@3: {}'.format(map_at_3))
    print('.' * 64)

    return step


def test(session, model, data, dataset_handle):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    all_keyids = []
    all_predictions = []

    session.run(data['test_iterator'].initializer)

    while True:
        try:
            feeds = {
                data['dataset_handle']: dataset_handle,
            }

            fetch = {
                'keyids': data['keyids'],
                'logits': model['logits'],
            }

            if 'training' in model:
                feeds[model['training']] = False

            fetched = session.run(fetch, feed_dict=feeds)

            keyids = fetched['keyids']

            predictions = np.argsort(fetched['logits'], axis=1)

            predictions = predictions[:, -1:-4:-1]

            all_keyids.append(keyids)
            all_predictions.append(predictions)
        except tf.errors.OutOfRangeError:
            break

    all_keyids = np.concatenate(all_keyids, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    if FLAGS.result_zip_path is not None:
        with gzip.open(
                FLAGS.result_zip_path, mode='wt', encoding='utf-8') as zf:
            writer = csv.writer(zf, lineterminator='\n')

            writer.writerow(['key_id', 'word'])

            for idx, keyid in enumerate(all_keyids):
                predictions = [dataset.index_to_label[all_predictions[idx, i]] for i in range(3)]

                predictions = ' '.join(predictions)

                writer.writerow([keyid, predictions])


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    data = build_dataset()

    model = build_model(data)

    summaries = build_summaries(model)

    reporter = tf.summary.FileWriter(FLAGS.logs_path)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        step = session.run(model['step'])

        # NOTE: exclude log which does not happend yet :)
        reporter.add_session_log(
            tf.SessionLog(status=tf.SessionLog.START), global_step=step)

        # NOTE: initialize dataset iterator
        train_record_paths = tf.gfile.ListDirectory(FLAGS.train_dir_path)
        train_record_paths = \
            [os.path.join(FLAGS.train_dir_path, n) for n in train_record_paths]

        session.run(
            data['train_iterator'].initializer,
            feed_dict={data['train_record_paths']: train_record_paths})
        session.run(data['valid_iterator'].initializer)

        # NOTE: generate handles for switching dataset
        train_handle = session.run(data['train_iterator'].string_handle())
        valid_handle = session.run(data['valid_iterator'].string_handle())
        test_handle = session.run(data['test_iterator'].string_handle())

        while True:
            step = train(
                session,
                step,
                model,
                data,
                train_handle,
                summaries,
                reporter)

            step = valid(
                session,
                step,
                model,
                data,
                valid_handle,
                summaries,
                reporter)

            if step >= FLAGS.stop_at_step:
                break

        test(session, model, data, test_handle)

        if not FLAGS.save_checkpoint:
            return

        target_ckpt_path = os.path.join(FLAGS.ckpt_path, 'model.ckpt')

        tf.train.Saver().save(
            session, target_ckpt_path, write_meta_graph=False)

    # NOTE: save meta, replace dataset with placeholder
    tf.reset_default_graph()

    data['images'] = tf.placeholder(
        tf.float32,
        shape=[None, FLAGS.image_size, FLAGS.image_size, 1],
        name='images')
    data['strokes'] = tf.placeholder(
        tf.float32,
        shape=[None, None, 3],
        name='strokes')
    data['lengths'] = tf.placeholder(
        tf.int32,
        shape=[None],
        name='lengths')
    data['labels'] = None

    model = build_model(data)

    tf.train.Saver().export_meta_graph(target_ckpt_path + '.meta')


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('train_dir_path', None, '')
    tf.app.flags.DEFINE_string('valid_tfr_path', None, '')
    tf.app.flags.DEFINE_string('ckpt_path', None, '')
    tf.app.flags.DEFINE_string('logs_path', None, '')
    tf.app.flags.DEFINE_string('model', None, '')
    tf.app.flags.DEFINE_string('learning_rate_policy', None, '')
    tf.app.flags.DEFINE_string('dataset_rotate_policy', None, '')

    tf.app.flags.DEFINE_string('test_tfr_path', None, '')
    tf.app.flags.DEFINE_string('result_zip_path', None, '')

    tf.app.flags.DEFINE_boolean('save_checkpoint', False, '')
    tf.app.flags.DEFINE_boolean('train_on_recognized', False, '')

    tf.app.flags.DEFINE_integer('image_size', 28, '')
    tf.app.flags.DEFINE_integer('batch_size', 100, '')

    tf.app.flags.DEFINE_integer('stop_at_step', 1000, '')

    tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001, '')

    tf.app.run()

