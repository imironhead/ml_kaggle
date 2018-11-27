"""
https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/applications/resnet.py
"""
import functools

import tensorflow as tf


def residual_block(tensors, stride, activation, normalization, initializer):
    """
    """
    identity = tensors

    num_filters_in = tensors.shape[-1]
    num_filters_out = tensors.shape[-1] * stride

    tensors = tf.layers.conv2d(
        tensors,
        filters=num_filters_out,
        kernel_size=3,
        strides=stride,
        padding='same',
        activation=None,
        use_bias=True,
        kernel_initializer=initializer)

    tensors = normalization(tensors)

    tensors = activation(tensors)

    tensors = tf.layers.conv2d(
        tensors,
        filters=num_filters_out,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=None,
        use_bias=True,
        kernel_initializer=initializer)

    if num_filters_in != num_filters_out:
        identity = tf.layers.conv2d(
            identity,
            filters=num_filters_out,
            kernel_size=1,
            strides=stride,
            padding='same',
            activation=None,
            use_bias=True,
            kernel_initializer=initializer)

        identity = normalization(identity)

    tensors = identity + tensors

    tensors = normalization(tensors)

    tensors = activation(tensors)

    return tensors


def basic_block(tensors, num_blocks, activation, normalization, initializer):
    """
    """
    block = functools.partial(
        residual_block,
        activation=activation,
        normalization=normalization,
        initializer=initializer)

    for i in range(num_blocks):
        tensors = block(tensors, stride=(2 if i == 0 else 1))

    return tensors


def build_model(images, strokes, lengths, labels, training):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    normalization = functools.partial(
        tf.contrib.layers.batch_norm, is_training=training)

    activation = tf.nn.relu

    block = functools.partial(
        basic_block,
        activation=activation,
        normalization=normalization,
        initializer=initializer)

    tensors = images

    # NOTE: original resnet 34
#   tensors = tf.layers.conv2d(
#       tensors,
#       filters=64,
#       kernel_size=7,
#       strides=2,
#       padding='same',
#       activation=None,
#       use_bias=True,
#       kernel_initializer=initializer)

#   tensors = normalization(tensors)

#   tensors = activation(tensors)

#   tensors = tf.nn.max_pool(
#       tensors, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    # NOTE: reduce to low resolution directly
    tensors = tf.nn.avg_pool(
        tensors, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    tensors = tf.layers.conv2d(
        tensors,
        filters=64,
        kernel_size=5,
        strides=2,
        padding='same',
        activation=None,
        use_bias=True,
        kernel_initializer=initializer)

    tensors = normalization(tensors)

    tensors = activation(tensors)

    # NOTE: back to original resnet
    tensors = block(tensors, 3)
    tensors = block(tensors, 4)
    tensors = block(tensors, 6)
    tensors = block(tensors, 3)

    tensors = tf.reduce_mean(tensors, axis=[1, 2])

    # NOTE: flatten for fc
    tensors = tf.layers.flatten(tensors)

    # NOTE:
    logits = tf.layers.dense(
        tensors,
        units=340,
        activation=None,
        use_bias=True,
        kernel_initializer=initializer,
        name='final')

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels,
        logits,
        reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

    return {
        'images': images,
        'labels': labels,
        'strokes': strokes,
        'lengths': lengths,
        'logits': logits,
        'loss': loss,
    }
