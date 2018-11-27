"""
"""
import functools

import tensorflow as tf


def block_mobilenets_v2(
        tensors,
        stride,
        filters,
        expansion,
        name,
        activation,
        normalization,
        initializer):
    """
    """
    num_filters_in = tensors.shape[-1]

    identity = tensors

    tensors = tf.layers.conv2d(
        tensors,
        filters=num_filters_in * expansion,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=None,
        use_bias=False,
        kernel_initializer=initializer)

    tensors = normalization(tensors)

    tensors = activation(tensors)

    filter_weights = tf.get_variable(
        'depthwise_conv2d_{}'.format(name),
        [3, 3, num_filters_in * expansion, 1],
        initializer=initializer,
        dtype=tf.float32)

    tensors = tf.nn.depthwise_conv2d(
        tensors,
        filter_weights,
        strides=[1, stride, stride, 1],
        padding='SAME',
        rate=[1, 1])

    tensors = normalization(tensors)

    tensors = activation(tensors)

    tensors = tf.layers.conv2d(
        tensors,
        filters=filters,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=None,
        use_bias=False,
        kernel_initializer=initializer)

    tensors = normalization(tensors)

    if stride != 1 or num_filters_in != filters:
        identity = tf.layers.conv2d(
            identity,
            filters=filters,
            kernel_size=1,
            strides=stride,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=initializer)

        identity = normalization(identity)

    tensors = tf.add(tensors, identity)

    tensors = activation(tensors)

    return tensors


def build_model(images, strokes, lengths, labels, training):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    normalization = functools.partial(
        tf.contrib.layers.batch_norm, is_training=training)

    activation = tf.nn.relu

    block = functools.partial(
        block_mobilenets_v2,
        expansion=1,
        activation=activation,
        normalization=normalization,
        initializer=initializer)

    tensors = images

    tensors = tf.layers.conv2d(
        tensors,
        filters=16,
        kernel_size=5,
        strides=2,
        padding='same',
        activation=None,
        use_bias=True,
        kernel_initializer=initializer)

    tensors = normalization(tensors)

    tensors = activation(tensors)

    tensors = block(tensors, stride=2, filters=32, name='b0')
    tensors = block(tensors, stride=1, filters=32, name='b1')
    tensors = block(tensors, stride=2, filters=64, name='b2')
    tensors = block(tensors, stride=1, filters=64, name='b3')
    tensors = block(tensors, stride=2, filters=128, name='b4')
    tensors = block(tensors, stride=1, filters=128, name='b5')
    tensors = block(tensors, stride=1, filters=128, name='b6')
    tensors = block(tensors, stride=2, filters=256, name='b7')
    tensors = block(tensors, stride=1, filters=256, name='b8')
    tensors = block(tensors, stride=1, filters=256, name='b9')
    tensors = block(tensors, stride=2, filters=512, name='b10')
    tensors = block(tensors, stride=1, filters=512, name='b11')
    tensors = block(tensors, stride=1, filters=512, name='b12')
    tensors = block(tensors, stride=1, filters=512, name='b13')
    tensors = block(tensors, stride=1, filters=512, name='b14')

    tensors = tf.reduce_mean(tensors, axis=[1, 2])

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

