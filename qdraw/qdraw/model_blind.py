
"""
"""
import functools

import tensorflow as tf


def mobilenet_block(
        tensors,
        filters,
        stride,
        expansion,
        activation,
        normalization,
        initializer,
        scope_name):
    """
    """
    with tf.variable_scope(scope_name):
        in_channels = tensors.shape[-1]

        filter_weights = tf.get_variable(
            'depthwise_conv_filter',
            [3, 3, in_channels, expansion],
            trainable=True,
            initializer=initializer,
            dtype=tf.float32)

        tensors = tf.nn.depthwise_conv2d(
            tensors,
            filter_weights,
            strides=[1, stride, stride, 1],
            padding='SAME',
            rate=[1, 1],
            name='depthwise_conv')

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
            kernel_initializer=initializer,
            name='pointwise_conv')

        tensors = normalization(tensors)

        tensors = activation(tensors)

    return tensors


def mobilenet_block_v2(
        tensors,
        filters,
        stride,
        expansion,
        activation,
        normalization,
        initializer,
        scope_name):
    """
    """
    num_channels_in = tensors.shape[-1]

    source_tensors = tensors

    with tf.variable_scope(scope_name):
        tensors = tf.layers.conv2d(
            tensors,
            filters=num_channels_in * expansion,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=initializer,
            name='head')

        tensors = normalization(tensors)

        tensors = activation(tensors)

        filter_weights = tf.get_variable(
            'depthwise_conv_filter',
            [3, 3, num_channels_in * expansion, 1],
            initializer=initializer,
            dtype=tf.float32)

        tensors = tf.nn.depthwise_conv2d(
            tensors,
            filter_weights,
            strides=[1, stride, stride, 1],
            padding='SAME',
            rate=[1, 1],
            name='depthwise_conv')

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
            kernel_initializer=initializer,
            name='tail')

        tensors = normalization(tensors)

        if stride == 1 and num_channels_in == filters:
            tensors = tf.add(tensors, source_tensors)

    return tensors


def resnext_block(
        tensors,
        stride,
        num_groups,
        activation,
        normalization,
        initializer,
        scope_name):
    """
    """
    num_channels_in = tensors.shape[-1]

    identity_tensors = tensors

    with tf.variable_scope(scope_name):
        bottleneck_filters = num_channels_in // 2

        tensors = tf.layers.conv2d(
            tensors,
            filters=bottleneck_filters,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=initializer,
            name='conv_1x1_1')

        tensors = normalization(tensors)

        tensors = activation(tensors)

        group_size = bottleneck_filters // num_groups

        filter_weights = tf.get_variable(
            'depthwise_conv_filter',
            [3, 3, bottleneck_filters, group_size * stride],
            initializer=initializer,
            dtype=tf.float32)

        tensors = tf.nn.depthwise_conv2d(
            tensors,
            filter_weights,
            strides=[1, stride, stride, 1],
            padding='SAME',
            rate=[1, 1],
            name='depthwise_conv')

        tensors = tf.reshape(
            tensors,
            [-1] + tensors.shape.as_list()[1:3] + [num_groups, group_size, group_size])

        tensors = tf.reduce_sum(tensors, axis=4)

        tensors = tf.reshape(
            tensors,
            [-1] + tensors.shape.as_list()[1:3] + [bottleneck_filters * stride])

        tensors = normalization(tensors)

        tensors = activation(tensors)

        tensors = tf.layers.conv2d(
            tensors,
            filters=bottleneck_filters * stride * 2,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=initializer,
            name='conv_1x1_2')

        tensors = normalization(tensors)

        if stride == 2:
            identity_tensors = tf.nn.avg_pool(
                identity_tensors, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

            identity_tensors = tf.pad(
                identity_tensors,
                [[0, 0], [0, 0], [0, 0], [num_channels_in // 2] * 2])

        tensors = tensors + identity_tensors

        tensors = activation(tensors)

    return tensors


def residual_block(
        tensors,
        filters,
        activation,
        normalization,
        initializer,
        scope_name):
    """
    """
    identity = tensors

    with tf.variable_scope(scope_name):
        tensors = tf.layers.conv2d(
            tensors,
            filters=filters,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=initializer,
            name='conv_1')

        tensors = normalization(tensors)

        tensors = activation(tensors)

        tensors = tf.layers.conv2d(
            tensors,
            filters=filters,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=initializer,
            name='conv_2')

        tensors = normalization(tensors)

        tensors = identity + tensors

        tensors = activation(tensors)

    return tensors


def build_model(images, strokes, lengths, labels, training):
    """
    """
    activation = tf.nn.relu

    normalization = functools.partial(
        tf.contrib.layers.batch_norm, is_training=training)

    initializer = tf.truncated_normal_initializer(stddev=0.02)

    block = functools.partial(
        resnext_block,
        num_groups=16,
        activation=activation,
        normalization=normalization,
        initializer=initializer)

    tensors = images

    tensors = tf.layers.conv2d(
        tensors,
        filters=64,
        kernel_size=3,
        strides=2,
        padding='same',
        activation=None,
        use_bias=False,
        kernel_initializer=initializer,
        name='conv_in')

    tensors = normalization(tensors)

    tensors = activation(tensors)

    tensors = block(tensors, 1, scope_name='1')
    tensors = block(tensors, 2, scope_name='2')
    tensors = block(tensors, 1, scope_name='3')
    tensors = block(tensors, 2, scope_name='4')
    tensors = block(tensors, 1, scope_name='5')
    tensors = block(tensors, 2, scope_name='6')
    tensors = block(tensors, 1, scope_name='7')
    tensors = block(tensors, 1, scope_name='8')
    tensors = block(tensors, 1, scope_name='9')

    tensors = tf.reduce_mean(tensors, axis=[1, 2])

    # NOTE: flatten for fc
    tensors = tf.layers.flatten(tensors)

    tensors = tf.layers.dropout(tensors, rate=1e-3, training=training)

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

