"""
"""
import csv
import gzip
import os
import time

import numpy as np
import tensorflow as tf

import qdraw.dataset as dataset
import qdraw.dataset_iterator as dataset_iterator


def build_model(data):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    if FLAGS.model == 'mobilenets':
        import qdraw.model_mobilenets as chosen_model
    elif FLAGS.model == 'mobilenets_v2':
        import qdraw.model_mobilenets_v2 as chosen_model
    elif FLAGS.model == 'blind':
        import qdraw.model_blind as chosen_model
    elif FLAGS.model == 'null':
        import qdraw.model_null as chosen_model

    with tf.device('/cpu:0'):
        # NOTE:
        step = tf.train.get_or_create_global_step()

        # NOTE:
        training = tf.placeholder(shape=[], dtype=tf.bool)

        # NOTE:
        learning_rate = tf.placeholder(shape=[], dtype=tf.float32)

        # NOTE:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    losses = []
    tower_grads = []

    for i in range(FLAGS.num_gpus):
        keyids, images, strokes, lengths, recognized, labels = \
            data['iterator'].get_next()

        with tf.variable_scope('ngpus', reuse=tf.AUTO_REUSE):
            with tf.name_scope('qdraw_{}'.format(i)) as scope:

                device_setter = tf.train.replica_device_setter(
                    worker_device='/gpu:{}'.format(i), ps_device='/cpu:0', ps_tasks=1)

                with tf.device(device_setter):
                    model = chosen_model.build_model(
                        images, strokes, lengths, labels, training)

                    losses.append(model['loss'])

                    # NOTE: am I right?
                    update_ops = \
                        tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

                    with tf.control_dependencies(update_ops):
                        grads = optimizer.compute_gradients(model['loss'])

                    tower_grads.append(grads)

                    # NOTE: all data go through a single gpu for inference
                    if i == 0:
                        keyids = keyids
                        labels = model['labels']
                        logits = model['logits']

    with tf.device('/cpu:0'):
        # NOTE: aggregate losses
        loss = tf.stack(losses)
        loss = tf.reduce_mean(loss, 0)

        # NOTE: aggregate gradients
        average_grads = []

        for grad_and_vars in zip(*tower_grads):
            grad = tf.stack([g for g, v in grad_and_vars])
            grad = tf.reduce_mean(grad, 0)

            average_grads.append((grad, grad_and_vars[0][1]))

        # NOTE:
        op_apply_gradient = \
            optimizer.apply_gradients(average_grads, global_step=step)

    return {
        'step': step,
        'loss': loss,
        'keyids': keyids,
        'labels': labels,
        'logits': logits,
        'recognized': recognized,
        'training': training,
        'learning_rate': learning_rate,
        'optimizer': op_apply_gradient,
        'dataset_handle': data['dataset_handle'],
    }


def build_summaries(model):
    """
    """
    return {
        'train_loss': tf.summary.scalar('train_loss', model['loss']),
    }


def train(session, model, dataset_handle, summaries, reporter):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    # NOTE: decay policy?
    step = session.run(model['step'])

    learning_rate = FLAGS.initial_learning_rate * (0.5 ** (step // 100000))

    feeds = {
        model['dataset_handle']: dataset_handle,
        model['learning_rate']: learning_rate,
        model['training']: True,
    }

    fetch = {
        'optimizer': model['optimizer'],
        'loss': model['loss'],
        'step': model['step'],
        'summary': summaries['train_loss'],
    }

    fetched = session.run(fetch, feed_dict=feeds)

    reporter.add_summary(fetched['summary'], fetched['step'])

    if fetched['step'] % 1000 == 0:
        print('loss[{}]: {}'.format(fetched['step'], fetched['loss']))

    return fetched['step']


def valid(session, model, data, dataset_handle, summaries, reporter):
    """
    """
    step = session.run(model['step'])

    if step % 10000 != 0:
        return step

    session.run(data['valid_iterator'].initializer)

    losses = 0.0
    num_images = 0
    aps = [0.0, 0.0, 0.0]

    num_recognized_correct = 0

    while True:
        try:
            feeds = {
                model['dataset_handle']: dataset_handle,
                model['training']: False,
            }

            fetch = {
                'loss': model['loss'],
                'labels': model['labels'],
                'logits': model['logits'],
                'recognized': model['recognized'],
            }

            fetched = session.run(fetch, feed_dict=feeds)

            losses = losses + fetched['loss']

            labels = fetched['labels']

            logits = np.argsort(fetched['logits'], axis=1)

            logits = logits[:, -1:-4:-1]

            num_images += logits.shape[0]

            for i in range(3):
                aps[i] += np.sum(logits[:, i] == labels) / float(i + 1)

            correct = (logits[:, 0] == labels)
            recognized = (fetched['recognized'] == 1)

            num_recognized_correct += np.sum(correct == recognized)
        except tf.errors.OutOfRangeError:
            break

    map_at_3 = sum(aps) / float(num_images)

    # NOTE: validation loss & map summary
    losses = losses / float(num_images)

    summaries = [
        tf.Summary.Value(tag='valid_loss', simple_value=losses),
        tf.Summary.Value(tag='map', simple_value=map_at_3),
    ]

    summaries = tf.Summary(value=summaries)

    reporter.add_summary(summaries, step)

    print('validation at step: {}'.format(step))
    print('on recognized: {}'.format(num_recognized_correct / num_images))
    print('map_1: {}'.format(aps[0] / num_images))
    print('map_2: {}'.format(aps[1] / num_images))
    print('map_3: {}'.format(aps[2] / num_images))
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
                model['dataset_handle']: dataset_handle,
                model['training']: False,
            }

            fetch = {
                'keyids': model['keyids'],
                'logits': model['logits'],
            }

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

        print('done: {}'.format(FLAGS.result_zip_path))


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    data = dataset_iterator.build_dataset(
        batch_size=FLAGS.batch_size,
        image_size=FLAGS.image_size,
        valid_dir_path=FLAGS.valid_dir_path,
        test_dir_path=FLAGS.test_dir_path,
        train_on_recognized=FLAGS.train_on_recognized)

    model = build_model(data)

    summaries = build_summaries(model)

    reporter = tf.summary.FileWriter(FLAGS.logs_path)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

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

        ts = time.clock()

        while True:
            step = train(session, model, train_handle, summaries, reporter)

            step = valid(session, model, data, valid_handle, summaries, reporter)

            if step % 1000 == 0:
                t = time.clock()

                print('[gpux{}] 1000 steps take {} seconds'.format(
                    FLAGS.num_gpus, int(t - ts)))

                ts = t

            if step >= FLAGS.stop_at_step:
                break

        test(session, model, data, test_handle)


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('train_dir_path', None, '')
    tf.app.flags.DEFINE_string('valid_dir_path', None, '')
    tf.app.flags.DEFINE_string('ckpt_path', None, '')
    tf.app.flags.DEFINE_string('logs_path', None, '')
    tf.app.flags.DEFINE_string('model', None, '')
    tf.app.flags.DEFINE_string('learning_rate_policy', None, '')
    tf.app.flags.DEFINE_string('dataset_rotate_policy', None, '')

    tf.app.flags.DEFINE_string('test_dir_path', None, '')
    tf.app.flags.DEFINE_string('result_zip_path', None, '')

    tf.app.flags.DEFINE_boolean('save_checkpoint', False, '')
    tf.app.flags.DEFINE_boolean('train_on_recognized', False, '')

    tf.app.flags.DEFINE_integer('num_gpus', 1, '')
    tf.app.flags.DEFINE_integer('image_size', 28, '')
    tf.app.flags.DEFINE_integer('batch_size', 100, '')

    tf.app.flags.DEFINE_integer('stop_at_step', 1000, '')

    tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001, '')

    tf.app.run()

