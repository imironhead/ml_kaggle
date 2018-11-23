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
    if FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    elif FLAGS.optimizer == 'nesterov':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=0.9,
            use_nesterov=True)

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

        'swa': [],
    }

    with tf.control_dependencies(update_ops):
        gradients_and_vars = optimizer.compute_gradients(model['loss'])

    # NOTE: helper function to create a placeholder for a variable
    def placeholder(g):
        return tf.placeholder(shape=g.shape, dtype=g.dtype)

    # NOTE: if cyclic_batch_size_multiplier_tail is great than 1, we will do
    #       gradients aggregation later
    if FLAGS.cyclic_batch_size_multiplier_tail > 1:
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
        for variable in tf.trainable_variables():
            ph = placeholder(variable)

            new_model['swa'].append({
                'variable': variable,
                'placeholder': ph,
                'var_op': tf.assign(variable, ph),
                'amount': 0.0,
                'swa_weights': 0.0,
                'tmp_weights': 0.0})

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
        beta = (FLAGS.cyclic_num_steps - 1) // (scale_tail - scale_head + 1)

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
        # NOTE: FLAGS.cyclic_batch_size_multiplier_tail <= 1, need no
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
        for v in model['swa']:
            weights = session.run(v['variable'])

            v['swa_weights'] = \
                (v['swa_weights'] * v['amount'] + weights) / (v['amount'] + 1.0)

            v['amount'] += 1.0

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

    if step % FLAGS.valid_cycle != 0:
        return

    # NOTE: Averaging Weights Leads to Wider Optima and Better
    #       Generalization
    #       catch trained variables and replace them with swa
    #       update running average of trainable variables for validation
    for v in model['swa']:
        v['tmp_weights'] = session.run(v['variable'])

        session.run(
            v['var_op'], feed_dict={v['placeholder']: v['swa_weights']})

    session.run(data['valid_iterator'].initializer)

    losses = 0.0
    logits = []
    labels = []
    recognized = []

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

            losses = losses + fetched['loss'] * fetched['labels'].shape[0]

            logits.append(fetched['logits'])

            labels.append(fetched['labels'])

            recognized.append(fetched['recognized'])
        except tf.errors.OutOfRangeError:
            break

    # NOTE: a method to evaluate on a dataset
    def evaluate(softmax, labels, recognized, name, reporter, step):
        """
        """
        num_images = softmax.shape[0]

        predictions = np.argsort(softmax, axis=1)
        predictions = predictions[:, -1:-4:-1]

        correct = (predictions[:, 0] == labels)
        recognized = (recognized == 1)

        accuracy_on_recognized = \
            np.sum(correct == recognized) / np.sum(recognized)

        map_1 = np.sum(predictions[:, 0] == labels) / (1.0 * num_images)
        map_2 = np.sum(predictions[:, 1] == labels) / (2.0 * num_images)
        map_3 = np.sum(predictions[:, 2] == labels) / (3.0 * num_images)

        map_all = map_1 + map_2 + map_3

        summaries = [
            tf.Summary.Value(tag='{}_map3'.format(name), simple_value=map_all),
        ]

        summaries = tf.Summary(value=summaries)

        reporter.add_summary(summaries, step)

        tf.logging.info('{} step {}'.format(name, step))
        tf.logging.info(
            'accuracy on recognized: {}'.format(accuracy_on_recognized))
        tf.logging.info('map_1: {}'.format(map_1))
        tf.logging.info('map_2: {}'.format(map_2))
        tf.logging.info('map_3: {}'.format(map_3))
        tf.logging.info('map@3: {}'.format(map_all))
        tf.logging.info('-' * 64)

    # NOTE: concat matrix
    logits = np.concatenate(logits, axis=0)
    labels = np.concatenate(labels, axis=0)
    recognized = np.concatenate(recognized, axis=0)

    # NOTE: do softmax on logits
    temp = np.exp(logits)

    softmax = temp / np.sum(temp, axis=1, keepdims=True)

    # NOTE: loss validation loss
    loss = losses / logits.shape[0]

    summary = tf.Summary(
        value=[tf.Summary.Value(tag='valid_loss', simple_value=loss)])

    reporter.add_summary(summary, step)

    # NOTE: evaluate on whole validation set
    evaluate(softmax, labels, recognized, 'valid_all', reporter, step)

    # NOTE: evaluate on tta
    if FLAGS.tta_enable:
        num_groups = softmax.shape[0] // FLAGS.tta_num_samples_valid

        num_classes = softmax.shape[-1]

        softmax = np.reshape(
            softmax, [num_groups, FLAGS.tta_num_samples_valid, num_classes])

        softmax = np.sum(softmax, axis=0)

        labels = labels[:FLAGS.tta_num_samples_valid]

        recognized = recognized[:FLAGS.tta_num_samples_valid]

        evaluate(softmax, labels, recognized, 'valid_tta', reporter, step)

    # NOTE: Averaging Weights Leads to Wider Optima and Better
    #       Generalization
    #
    #       roll back training weights
    for v in model['swa']:
        session.run(
            v['var_op'], feed_dict={v['placeholder']: v['tmp_weights']})


def test(session, model, data, dataset_handle):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    if FLAGS.result_zip_path is None:
        return

    keyids = []
    logits = []

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

            keyids.append(fetched['keyids'])
            logits.append(fetched['logits'])
        except tf.errors.OutOfRangeError:
            break

    # NOTE: concat matrix
    keyids = np.concatenate(keyids, axis=0)
    logits = np.concatenate(logits, axis=0)

    # NOTE: do softmax on logits
    temp = np.exp(logits)

    softmax = temp / np.sum(temp, axis=1, keepdims=True)

    # NOTE: evaluate on tta
    if FLAGS.tta_enable:
        num_groups = softmax.shape[0] // FLAGS.tta_num_samples_test

        num_classes = softmax.shape[-1]

        softmax = np.reshape(
            softmax, [num_groups, FLAGS.tta_num_samples_test, num_classes])

        softmax = np.sum(softmax, axis=0)
    else:
        # NOTE: remove augmented part if tta is not enabled
        softmax = softmax[:FLAGS.tta_num_samples_test]

    keyids = keyids[:FLAGS.tta_num_samples_test]

    predictions = np.argsort(softmax, axis=1)
    predictions = predictions[:, -1:-4:-1]

    with gzip.open(FLAGS.result_zip_path, mode='wt', encoding='utf-8') as zf:
        writer = csv.writer(zf, lineterminator='\n')

        writer.writerow(['key_id', 'word'])

        for idx, keyid in enumerate(keyids):
            guess = [
                dataset.index_to_label[
                    predictions[idx, i]] for i in range(3)]

            guess = ' '.join(guess)

            writer.writerow([keyid, guess])

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
        #       Generalization
        #
        #       replace weights with swa weights
        for v in model['swa']:
            session.run(
                v['var_op'], feed_dict={v['placeholder']: v['swa_weights']})

        # NOTE: Averaging Weights Leads to Wider Optima and Better
        #       Generalization
        #
        #       update batch normalization
        feeds = {
            model['dataset_handle']: train_handle,
            model['training']: True,
        }

        for _ in range(100):
            session.run(model['gradients_result'], feed_dict=feeds)

        # NOTE: final test
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

    # NOTE: do validation every FLAGS.valid_cycle steps
    tf.app.flags.DEFINE_integer('valid_cycle', 10000, '')

    # NOTE: optimizer
    tf.app.flags.DEFINE_string('optimizer', 'adam', '')

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

    # NOTE: test time augmentation
    tf.app.flags.DEFINE_boolean('tta_enable', False, '')

    # NOTE: if FLAGS.tta_enable is true, apply tta to validation and test set.
    #
    #       number of validation samples would be FLAGS.tta_num_samples_valid
    #       and not be shuffled.
    #
    #       number of testing samples would be FLAGS.tta_num_samples_test and
    #       not be shuffled
    tf.app.flags.DEFINE_integer('tta_num_samples_valid', 0, '')
    tf.app.flags.DEFINE_integer('tta_num_samples_test', 0, '')

    # NOTE: stochastic weight averaging (SWA)
    tf.app.flags.DEFINE_boolean('swa_enable', False, '')

    # NOTE:
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run()
