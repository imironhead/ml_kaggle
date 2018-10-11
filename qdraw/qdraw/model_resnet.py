"""
"""
import numpy as np
import tensorflow as tf



def group_norm(tensors, num_channels, num_groups, scope_name):
    """
    """
    with tf.variable_scope(scope_name):
        gamma = tf.get_variable(
            'gamma',
            [1, num_channels, 1, 1],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.5))

        beta = tf.get_variable(
            'beta',
            [1, num_channels, 1, 1],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.5))

        shape = tf.shape(tensors)

        n, c, h, w = shape[0], shape[1], shape[2], shape[3]

        tensors = tf.reshape(
            tensors, [n, num_groups, num_channels // num_groups, h, w])

        mean, var = tf.nn.moments(tensors, [2, 3, 4], keep_dims=True)

        tensors = (tensors - mean) / tf.sqrt(var + 1e-5)

        tensors = tf.reshape(tensors, [n, c, h, w])

        return tensors * gamma + beta


def residual_block(tensors, scope_name):
    """
    """
    identity = tensors

    initializer = tf.truncated_normal_initializer(stddev=0.02)

    with tf.variable_scope(scope_name):
        tensors = tf.layers.conv2d(
            tensors,
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            data_format='channels_first',
            activation=tf.nn.relu,
            use_bias=False,
            kernel_initializer=initializer,
            name='conv_1')

        tensors = group_norm(tensors, 64, 32, scope_name='gnorm')

        tensors = tf.layers.conv2d(
            tensors,
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            data_format='channels_first',
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=initializer,
            name='conv_2')

    return identity + tensors


def build_model(images, labels):
    """
    images: [None, 28, 28, 1] float32
    labels: [None], 340 categories
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    tensors = tf.transpose(images, perm=[0, 3, 1, 2])

    tensors = tf.layers.conv2d(
        tensors,
        filters=64,
        kernel_size=3,
        strides=2,
        padding='same',
        data_format='channels_first',
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=initializer,
        name='conv_1')

    for i in range(5):
        tensors = residual_block(tensors, scope_name='res_{}'.format(i))

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

    logits = tf.identity(logits, name='logits')

    # NOTE: build a simplier model without traning op
    if labels is None:
        return {
            'images': images,
            'logits': logits,
        }

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
        'logits': logits,
        'loss': loss,
        'step': step,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
    }

