"""
"""
import numpy as np
import ore
import os
import tensorflow as tf

from model import Model


tf.app.flags.DEFINE_boolean('predict', False, 'set to true to predict.')
tf.app.flags.DEFINE_string('checkpoint_dir', None, 'path to checkpoint_dir.')
tf.app.flags.DEFINE_string('log_dir', None, 'log_dir for tensorboard.')
tf.app.flags.DEFINE_string('out_dir', './', 'path to predict result')


def next_batch(reader, size):
    """
    """
    images, labels, is_new_batch = reader.next_batch(size, one_hot=True)

    images = 2.0 * (images - 1.0)

    # pad to 32 * 32 images with -1.0
    images = np.pad(
        images,
        ((0, 0), (2, 2), (2, 2), (0, 0)),
        'constant',
        constant_values=(-1.0, -1.0))

    return images, labels, is_new_batch


def predict():
    """
    """
    print('predicting')

    all_labels = []

    model = Model()

    reader = ore.DeterministicReader(ore.DATASET_KAGGLE_MNIST_TEST)

    while True:
        images, labels, is_new_batch = next_batch(reader, 128)

        if is_new_batch and len(all_labels) > 0:
            break

        labels = model.predict(images)

        all_labels.extend(labels)

    path_result = os.path.join(tf.app.flags.FLAGS.out_dir, 'result.txt')

    with open(path_result, 'wb') as output:
        output.write('ImageId,Label\n')

        for i, l in enumerate(all_labels):
            output.write('{},{}\n'.format(i + 1, l))


def train():
    """
    """
    print('training')

    model = Model()

    reader = ore.RandomReader(ore.DATASET_KAGGLE_MNIST_TRAINING)

    while True:
        images, labels, is_new_batch = next_batch(reader, 128)

        loss, step = model.train(images, labels)

        print('loss: {}'.format(loss))

        if step % 1000 == 0:
            model.save_checkpoint()


def main(_):
    """
    """
    if tf.app.flags.FLAGS.predict:
        predict()
    else:
        train()


if __name__ == '__main__':
    tf.app.run()
