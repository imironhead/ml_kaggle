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

    # NOTE: choose model
    if FLAGS.model == 'mobilenets':
        import qdraw.model_mobilenets as chosen_model
    elif FLAGS.model == 'mobilenets_v2':
        import qdraw.model_mobilenets_v2 as chosen_model
    elif FLAGS.model == 'blind':
        import qdraw.model_blind as chosen_model
    elif FLAGS.model == 'null':
        import qdraw.model_null as chosen_model

    # NOTE:
    step = tf.train.get_or_create_global_step()

    # NOTE:
    training = tf.placeholder(shape=[], dtype=tf.bool)

    # NOTE:
    learning_rate = tf.placeholder(shape=[], dtype=tf.float32)

    # NOTE:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    #
    keyids, images, strokes, lengths, recognized, labels = \
        data['iterator'].get_next()

    model = chosen_model.build_model(
        images, strokes, lengths, labels, training)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    new_model = {
        'keyids': keyids,
        'images': images,
        'strokes': strokes,
        'lengths': lengths,
        'recognized': recognized,
        'labels': labels,

        'step': step,
        'training': training,
        'learning_rate': learning_rate,
        'dataset_handle': data['dataset_handle'],

        'loss': model['loss'],
        'logits': model['logits'],
    }

    with tf.control_dependencies(update_ops):
        gradients_and_vars = optimizer.compute_gradients(model['loss'])

    # NOTE: if cyclic_batch_size_multiplier_tail is great than 1, we will do
    #       gradients aggregation later
    if FLAGS.cyclic_batch_size_multiplier_tail > 1:
        def placeholder(g):
            return tf.placeholder(shape=g.shape, dtype=g.dtype)

        # NOTE: an operator to collect computed gradients
        new_model['gradients_result'] = [g for g, v in gradients_and_vars]

        gradients_and_vars = \
            [(placeholder(g), v) for g, v in gradients_and_vars]

        # NOTE: an operator to feed manipulated gradients
        new_model['gradients_source'] = [g for g, v in gradients_and_vars]

    new_model['optimizer'] = \
        optimizer.apply_gradients(gradients_and_vars, global_step=step)

    # NOTE: Averaging Weights Leads to Wider Optima and Better
    #       Generalization, 3.2 batch normalization
    #       for updating training variables' running averaging
    if FLAGS.swa_enable:
        new_model['trainable_variables'] = tf.trainable_variables()

    return new_model


def train(session, model, dataset_handle, reporter):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    step = session.run(model['step'])

    # NOTE: learning rate interpolation for cyclic training
    lr_head = FLAGS.cyclic_learning_rate_head
    lr_tail = FLAGS.cyclic_learning_rate_tail

    alpha = (step % FLAGS.cyclic_num_steps) / (FLAGS.cyclic_num_steps - 1)

    learning_rate = lr_head + (lr_tail - lr_head) * alpha

    # NOTE: if cyclic_batch_size_multiplier_tail is great than 1, we want to do
    #       gradients aggregation
    if FLAGS.cyclic_batch_size_multiplier_tail > 1:
        # NOTE: batch multiplier interpolation for cyclic training
        scale_head = FLAGS.cyclic_batch_size_multiplier_head
        scale_tail = FLAGS.cyclic_batch_size_multiplier_tail

        # NOTE: assume FLAGS.cyclic_num_steps being far freater then scale
        beta = FLAGS.cyclic_num_steps // (scale_tail - scale_head + 1)

        batch_multiplier = scale_head + (step % FLAGS.cyclic_num_steps) // beta

        feeds = {
            model['dataset_handle']: dataset_handle,
            model['learning_rate']: learning_rate,
            model['training']: True,
        }

        fetch = [model['loss'], model['gradients_result']]

        all_gradients = []

        losses = 0.0

        for i in range(batch_multiplier):
            loss, gradients = session.run(fetch, feed_dict=feeds)

            losses += loss

            all_gradients.append(gradients)

        feeds = {
            model['learning_rate']: learning_rate,
        }

        for i, gradients_source in enumerate(model['gradients_source']):
            gradients = np.stack([g[i] for g in all_gradients], axis=0)

            feeds[gradients_source] = np.mean(gradients, axis=0)

        session.run(model['optimizer'], feed_dict=feeds)

        loss = losses / batch_multiplier
    else:
        # NOTE: FLAGS.cyclic_batch_size_multiplier_tail <= 1, do not need
        #       gradients aggregation
        feeds = {
            model['dataset_handle']: dataset_handle,
            model['learning_rate']: learning_rate,
            model['training']: True,
        }

        fetch = [model['loss'], model['optimizer']]

        loss, _ = session.run(fetch, feed_dict=feeds)

    step = session.run(model['step'])

    # NOTE: Averaging Weights Leads to Wider Optima and Better
    #       Generalization
    #       update running average of trainable variables
    if step % FLAGS.cyclic_num_steps == 0:
        pass

    # NOTE: training log
    summary = tf.Summary(
        value=[tf.Summary.Value(tag='train_loss', simple_value=loss)])

    reporter.add_summary(summary, step)

    if step % 1000 == 0:
        tf.logging.info('loss[{}]: {}'.format(step, loss))


def valid(session, model, data, dataset_handle, reporter):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    step = session.run(model['step'])

    if step % FLAGS.cyclic_num_steps != 0:
        return

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

    tf.logging.info('validation at step: {}'.format(step))
    tf.logging.info('on recognized: {}'.format(num_recognized_correct / num_images))
    tf.logging.info('map_1: {}'.format(aps[0] / num_images))
    tf.logging.info('map_2: {}'.format(aps[1] / num_images))
    tf.logging.info('map_3: {}'.format(aps[2] / num_images))
    tf.logging.info('map@3: {}'.format(map_at_3))
    tf.logging.info('.' * 64)


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
                predictions = [
                    dataset.index_to_label[
                        all_predictions[idx, i]] for i in range(3)]

                predictions = ' '.join(predictions)

                writer.writerow([keyid, predictions])

        tf.logging.info('done: {}'.format(FLAGS.result_zip_path))


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    data = dataset_iterator.build_dataset(
        batch_size=FLAGS.cyclic_batch_size,
        image_size=FLAGS.image_size,
        valid_dir_path=FLAGS.valid_dir_path,
        test_dir_path=FLAGS.test_dir_path,
        train_on_recognized=FLAGS.train_on_recognized)

    model = build_model(data)

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

        while True:
            train(session, model, train_handle, reporter)

            valid(session, model, data, valid_handle, reporter)

            step = session.run(model['step'])

            if step >= FLAGS.cyclic_num_steps * FLAGS.cyclic_num_cycles:
                break

        # NOTE: Averaging Weights Leads to Wider Optima and Better
        #       Generalization, 3.2 batch normalization
        #       if stochastic weight averaging (SWA) is enabled, feed some
        #       training into the network without applying gradients
        if FLAGS.swa_enable:
            pass

        test(session, model, data, test_handle)


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('model', None, '')

    tf.app.flags.DEFINE_string('ckpt_path', None, '')
    tf.app.flags.DEFINE_string('logs_path', None, '')

    tf.app.flags.DEFINE_string('train_dir_path', None, '')
    tf.app.flags.DEFINE_string('valid_dir_path', None, '')
    tf.app.flags.DEFINE_string('test_dir_path', None, '')
    tf.app.flags.DEFINE_string('result_zip_path', None, '')

    tf.app.flags.DEFINE_boolean('train_on_recognized', False, '')

    tf.app.flags.DEFINE_integer('image_size', 28, '')

    # NOTE: each cycle consists of cyclic_num_steps steps
    #       the model will be trained with cyclic_num_cycles cycles
    tf.app.flags.DEFINE_integer('cyclic_num_steps', 1, '')
    tf.app.flags.DEFINE_integer('cyclic_num_cycles', 1, '')

    # NOTE: in each cycle, linearly increase size of mini batch
    #       from cyclic_batch_size * cyclic_batch_size_multiplier_head
    #       to   cyclic_batch_size * cyclic_batch_size_multiplier_tail
    tf.app.flags.DEFINE_integer('cyclic_batch_size', 1, '')
    tf.app.flags.DEFINE_integer('cyclic_batch_size_multiplier_head', 1, '')
    tf.app.flags.DEFINE_integer('cyclic_batch_size_multiplier_tail', 1, '')

    # NOTE: in each cycle, linearly decrease learning rate
    #       from cyclic_learning_rate_head
    #       to   cyclic_learning_rate_tail
    tf.app.flags.DEFINE_float('cyclic_learning_rate_head', 0.0001, '')
    tf.app.flags.DEFINE_float('cyclic_learning_rate_tail', 0.0001, '')

    # NOTE: stochastic weight averaging (SWA)
    tf.app.flags.DEFINE_boolean('swa_enable', False, '')

    # NOTE:
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run()
