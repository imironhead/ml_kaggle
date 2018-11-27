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
    elif FLAGS.model == 'resnet':
        import qdraw.model_resnet as chosen_model
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

    # NOTE: helper function to create a placeholder for a variable
    def placeholder(g):
        return tf.placeholder(shape=g.shape, dtype=g.dtype)

    with tf.control_dependencies(update_ops):
        gradients_and_vars = optimizer.compute_gradients(model['loss'])

    # NOTE: if cyclic_batch_size_multiplier_max is great than 1, we will do
    #       gradients aggregation later
    # NOTE: an operator to collect computed gradients
    new_model['gradients_result'] = [g for g, v in gradients_and_vars]

    gradients_and_vars = [(placeholder(g), v) for g, v in gradients_and_vars]

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


def train(session, experiment):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    model = experiment['model']

    step = session.run(model['step'])

    # NOTE: learning rate interpolation for cyclic training
    lr_min = FLAGS.cyclic_learning_rate_min
    lr_max = FLAGS.cyclic_learning_rate_max

    alpha = (step % FLAGS.cyclic_num_steps) / (FLAGS.cyclic_num_steps - 1)

    learning_rate = lr_max + (lr_min - lr_max) * alpha

    # NOTE: feeds for training
    feeds = {
        model['dataset_handle']: experiment['data']['train_handle'],
        model['learning_rate']: learning_rate,
        model['training']: True,
    }

    # NOTE: if cyclic_batch_size_multiplier_max is great than 1, we want to do
    #       gradients aggregation
    # NOTE: batch multiplier interpolation for cyclic training
    scale_min = FLAGS.cyclic_batch_size_multiplier_min
    scale_max = FLAGS.cyclic_batch_size_multiplier_max

    # NOTE: assume FLAGS.cyclic_num_steps being far freater then scale
    beta = FLAGS.cyclic_num_steps // (scale_max - scale_min + 1)

    batch_multiplier = scale_min + (step % FLAGS.cyclic_num_steps) // beta

    all_gradients = []

    losses = 0.0

    # NOTE: compute gradients on nano batches
    for i in range(batch_multiplier):
        loss, gradients = session.run(
            [model['loss'], model['gradients_result']], feed_dict=feeds)

        losses += loss

        all_gradients.append(gradients)

    # NOTE: aggregate & apply gradients
    feeds = {
        model['learning_rate']: learning_rate,
    }

    for i, gradients_source in enumerate(model['gradients_source']):
        gradients = np.stack([g[i] for g in all_gradients], axis=0)

        feeds[gradients_source] = np.mean(gradients, axis=0)

    session.run(model['optimizer'], feed_dict=feeds)

    loss = losses / batch_multiplier

    step = session.run(model['step'])

    # NOTE: Averaging Weights Leads to Wider Optima and Better Generalization
    #
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

    experiment['reporter'].add_summary(summary, step)

    if step % 1000 == 0:
        tf.logging.info('loss[{}]: {}'.format(step, loss))


def valid(session, experiment):
    """
    """
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

    FLAGS = tf.app.flags.FLAGS

    model = experiment['model']

    reporter = experiment['reporter']

    step = session.run(model['step'])

    if step % FLAGS.valid_cycle != 0:
        return

    # NOTE: Averaging Weights Leads to Wider Optima and Better Generalization
    #
    #       catch trained variables and replace them with swa
    #       update running average of trainable variables for validation
    for v in model['swa']:
        v['tmp_weights'] = session.run(v['variable'])

        session.run(
            v['var_op'], feed_dict={v['placeholder']: v['swa_weights']})

    # NOTE: Averaging Weights Leads to Wider Optima and Better Generalization
    #
    #       update batch normalization
    feeds = {
        model['dataset_handle']: experiment['data']['train_handle'],
        model['training']: True,
    }

    for _ in range(100):
        session.run(model['gradients_result'], feed_dict=feeds)


    # NOTE: reset validation dataset
    session.run(experiment['data']['valid_iterator'].initializer)

    # NOTE: iterator of validation dataset only iterate the dataset once.
    losses = 0.0
    logits = []
    labels = []
    recognized = []

    feeds = {
        model['dataset_handle']: experiment['data']['valid_handle'],
        model['training']: False,
    }

    fetch = {
        'loss': model['loss'],
        'labels': model['labels'],
        'logits': model['logits'],
        'recognized': model['recognized'],
    }

    while True:
        try:
            fetched = session.run(fetch, feed_dict=feeds)

            losses = losses + fetched['loss'] * fetched['labels'].shape[0]

            logits.append(fetched['logits'])

            labels.append(fetched['labels'])

            recognized.append(fetched['recognized'])
        except tf.errors.OutOfRangeError:
            break

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


def test(session, experiment):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    if FLAGS.result_zip_path is None:
        return

    model = experiment['model']

    # NOTE: Averaging Weights Leads to Wider Optima and Better Generalization
    #
    #       replace weights with swa weights
    for v in model['swa']:
        session.run(
            v['var_op'], feed_dict={v['placeholder']: v['swa_weights']})

    # NOTE: Averaging Weights Leads to Wider Optima and Better Generalization
    #
    #       update batch normalization
    if model['swa']:
        feeds = {
            model['dataset_handle']: experiment['data']['train_handle'],
            model['training']: True,
        }

        for _ in range(100):
            session.run(model['gradients_result'], feed_dict=feeds)

    # NOTE: collect predictions
    keyids = []
    logits = []

    # NOTE: reset iterator of testing dataset
    session.run(experiment['data']['test_iterator'].initializer)

    feeds = {
        model['dataset_handle']: experiment['data']['test_handle'],
        model['training']: False,
    }

    fetch = {
        'keyids': model['keyids'],
        'logits': model['logits'],
    }

    while True:
        try:
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


def train_validate_test(session, experiment):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    # NOTE: build file writer to keep log
    experiment['reporter'] = tf.summary.FileWriter(FLAGS.logs_path)

    # NOTE: initialize all variables or load weights
    session.run(tf.global_variables_initializer())

    while True:
        train(session, experiment)

        valid(session, experiment)

        step = session.run(experiment['model']['step'])

        if step >= FLAGS.cyclic_num_steps * FLAGS.cyclic_num_cycles:
            break

    # NOTE: final test
    test(session, experiment)


def search_learning_rate(session, experiment):
    """
    """
    def train_one_step(session, experiment, lr):
        """
        """
        FLAGS = tf.app.flags.FLAGS

        model = experiment['model']

        # NOTE: feeds for training
        feeds = {
            model['dataset_handle']: experiment['data']['train_handle'],
            model['training']: True,
        }

        all_gradients = []

        # NOTE: compute gradients on nano batches
        for i in range(FLAGS.slr_batch_size_multiplier):
            gradients = session.run(model['gradients_result'], feed_dict=feeds)

            all_gradients.append(gradients)

        # NOTE: aggregate & apply gradients
        feeds = {model['learning_rate']: lr}

        for i, gradients_source in enumerate(model['gradients_source']):
            gradients = np.stack([g[i] for g in all_gradients], axis=0)

            feeds[gradients_source] = np.mean(gradients, axis=0)

        session.run(model['optimizer'], feed_dict=feeds)

    def valid_accuracy(session, experiment):
        """
        """
        FLAGS = tf.app.flags.FLAGS

        model = experiment['model']

        # NOTE: reset validation dataset
        session.run(experiment['data']['valid_iterator'].initializer)

        # NOTE: iterator of validation dataset only iterate the dataset once.
        logits = []
        labels = []

        feeds = {
            model['dataset_handle']: experiment['data']['valid_handle'],
            model['training']: False,
        }

        fetch = {
            'labels': model['labels'],
            'logits': model['logits'],
        }

        while True:
            try:
                fetched = session.run(fetch, feed_dict=feeds)

                logits.append(fetched['logits'])
                labels.append(fetched['labels'])
            except tf.errors.OutOfRangeError:
                break

        # NOTE: concat matrix
        logits = np.concatenate(logits, axis=0)
        labels = np.concatenate(labels, axis=0)

        predictions = np.argsort(logits, axis=1)
        predictions = predictions[:, -1]

        accuracy = np.sum(predictions == labels) / logits.shape[0]

        return accuracy

    # NOTE:
    FLAGS = tf.app.flags.FLAGS

    lr_min = FLAGS.slr_learning_rate_min
    lr_max = FLAGS.slr_learning_rate_max

    trials = []

    for index_trial in range(FLAGS.slr_num_trials):
        # NOTE: reset variables for next searching round
        session.run(tf.global_variables_initializer())

        if FLAGS.slr_random:
            lr = lr_min + (lr_max - lr_min) * np.random.random()
        else:
            lr = lr_min + (lr_max - lr_min) * (index_trial / FLAGS.slr_num_trials)

        # NOTE: train a little bits
        for _ in range(FLAGS.slr_num_steps):
            train_one_step(session, experiment, lr)

        accuracy = valid_accuracy(session, experiment)

        trials.append((lr, accuracy))

        tf.logging.info('acc [{}][{:.9f}]: {:.4f}'.format(index_trial, lr, accuracy))

    # NOTE: sort by learning rate
    trials.sort(lambda x: x[0])

    for lr, acc in trails:
        tf.logging.info('lr: {:.9f}, acc: {:.4f}'.format(lr, acc))


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    # NOTE: to keep information
    experiment = {}

    # NOTE: collect path of training data
    train_record_paths = tf.gfile.ListDirectory(FLAGS.train_dir_path)
    train_record_paths = \
        [os.path.join(FLAGS.train_dir_path, n) for n in train_record_paths]

    # NOTE: build dataset iterators
    experiment['data'] = dataset_iterator.build_dataset(
        batch_size=FLAGS.cyclic_batch_size,
        image_size=FLAGS.image_size,
        valid_dir_path=FLAGS.valid_dir_path,
        test_dir_path=FLAGS.test_dir_path,
        train_on_recognized=FLAGS.train_on_recognized)

    # NOTE: build model
    experiment['model'] = build_model(experiment['data'])

    with tf.Session() as session:
        # NOTE: initialize training data iterator
        session.run(
            experiment['data']['train_iterator'].initializer,
            feed_dict={
                experiment['data']['train_record_paths']: train_record_paths})

        # NOTE: generate handles for switching dataset
        experiment['data']['train_handle'] = \
            session.run(experiment['data']['train_iterator'].string_handle())
        experiment['data']['valid_handle'] = \
            session.run(experiment['data']['valid_iterator'].string_handle())
        experiment['data']['test_handle'] = \
            session.run(experiment['data']['test_iterator'].string_handle())

        if FLAGS.slr_num_trials > 0:
            search_learning_rate(session, experiment)
        else:
            train_validate_test(session, experiment)


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
    #       from cyclic_batch_size * cyclic_batch_size_multiplier_min
    #       to   cyclic_batch_size * cyclic_batch_size_multiplier_max
    tf.app.flags.DEFINE_integer('cyclic_batch_size', 1, '')
    tf.app.flags.DEFINE_integer('cyclic_batch_size_multiplier_min', 1, '')
    tf.app.flags.DEFINE_integer('cyclic_batch_size_multiplier_max', 1, '')

    # NOTE: in each cycle, linearly decrease learning rate
    #       from cyclic_learning_rate_max
    #       to   cyclic_learning_rate_min
    tf.app.flags.DEFINE_float('cyclic_learning_rate_min', 0.0001, '')
    tf.app.flags.DEFINE_float('cyclic_learning_rate_max', 0.0001, '')

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

    # NOTE: evaluate learning rate
    tf.app.flags.DEFINE_boolean('slr_random', False, '')
    tf.app.flags.DEFINE_integer('slr_num_trials', 0, '')
    tf.app.flags.DEFINE_integer('slr_num_steps', 0, '')
    tf.app.flags.DEFINE_integer('slr_batch_size_multiplier', 0, '')

    tf.app.flags.DEFINE_float('slr_learning_rate_min', 0.0001, '')
    tf.app.flags.DEFINE_float('slr_learning_rate_max', 0.0001, '')

    # NOTE:
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.app.run()
