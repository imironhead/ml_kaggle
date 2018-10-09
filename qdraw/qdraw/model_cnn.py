"""
"""
import numpy as np
import tensorflow as tf


def build_model(images, labels):
    """
    images: [None, 28, 28, 1] float32
    labels: [None], 340 categories
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    tensors = images

    for i in range(4):
        tensors = tf.layers.conv2d(
            tensors,
            filters=128,
            kernel_size=3,
            strides=1,
            padding='same',
            data_format='channels_last',
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=initializer,
            name='conv_{}'.format(i))

        # NOTE: norm ?

    # NOTE: position tag
    x = np.linspace(-1.0, 1.0, 28, dtype=np.float32)
    u, v = np.meshgrid(x, x)

    u = np.reshape(u, [1, 28, 28, 1])
    v = np.reshape(v, [1, 28, 28, 1])

    coordinates = np.concatenate([u, v], axis=-1)
    coordinates = tf.constant(coordinates)

    n = tf.shape(tensors)[0]

    coordinates = tf.tile(coordinates, [n, 1, 1, 1])

    tensors = tf.concat([tensors, coordinates], axis=-1)

    # NOTE: flatten for fc
    tensors = tf.layers.flatten(tensors)

    # NOTE
    for i, units in enumerate([2048, 2048, 512]):
        tensors = tf.layers.dense(
            tensors,
            units=units,
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=initializer,
            name='fc_{}'.format(i))

    # NOTE:
    logits = tf.layers.dense(
        tensors,
        units=340,
        activation=None,
        use_bias=False,
        kernel_initializer=initializer,
        name='final')

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels,
        logits,
        reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

    step = tf.train.get_or_create_global_step()

    learning_rate = tf.placeholder(shape=[], dtype=tf.float32)

    optimizer = tf.train \
        .AdamOptimizer(learning_rate=learning_rate) \
        .minimize(loss, global_step=step)

    return {
        'images': images,
        'labels': labels,
        'loss': loss,
        'step': step,
        'logits': logits,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
    }

