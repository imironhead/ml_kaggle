"""
"""
import functools

import tensorflow as tf


def block(
        tensors,
        filters,
        stride,
        expansion,
        survival_rate,
        activation,
        normalization,
        initializer,
        scope_name):
    """
    """
    num_filters_in = tensors.shape[-1]

    identity_tensors = tensors

    with tf.variable_scope(scope_name):
        tensors = tf.layers.conv2d(
            tensors,
            filters=num_filters_in * expansion,
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
            [3, 3, num_filters_in * expansion, 1],
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

        if stride == 1 and num_filters_in == filters:
            tensors = tf.add(tensors, identity_tensors)

            if survival_rate < 1.0:
                tensors = survival_rate * tensors

        tensors = activation(tensors)

    return tensors


def block_2(
        tensors,
        filters,
        stride,
        expansion,
        survival_rate,
        activation,
        normalization,
        initializer,
        scope_name):
    """
    """
    num_filters_in = tensors.shape[-1]

    identity_tensors = tensors

    with tf.variable_scope(scope_name):
        tensors = tf.layers.conv2d(
            tensors,
            filters=num_filters_in * expansion,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=initializer,
            name='head')

        tensors = normalization(tensors)

        tensors = activation(tensors)

        tensors = tf.layers.conv2d(
            tensors,
            filters=num_filters_in * expansion,
            kernel_size=3,
            strides=stride,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=initializer,
            name='mid')

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

        if stride == 1 and num_filters_in == filters:
            tensors = tf.add(tensors, identity_tensors)

            if survival_rate < 1.0:
                tensors = survival_rate * tensors

        tensors = activation(tensors)

    return tensors


def stochastic_block(
        tensors,
        stride,
        expansion,
        training,
        survival_rate,
        activation,
        normalization,
        initializer,
        scope_name):
    """
    """
    in_channels = tensors.shape[-1]

    filters = in_channels if stride == 1 else 2 * in_channels

    fn_block = functools.partial(
        block_2,
        tensors=tensors,
        filters=filters,
        stride=stride,
        expansion=expansion,
        activation=activation,
        normalization=normalization,
        initializer=initializer,
        scope_name=scope_name)

    # NOTE: pooling or keep alive
    if survival_rate >= 1.0 or stride != 1:
        return fn_block(survival_rate=1.0)

    # NOTE: take care of the issue on tf.cond
    #       https://github.com/tensorflow/tensorflow/issues/14699

    # NOTE: block for training
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        survived_tensors = fn_block(survival_rate=1.0)

        survival_roll = tf.random_uniform(
            shape=[], minval=0.0, maxval=1.0, name='survival')

        survive = tf.less(survival_roll, survival_rate)

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        test_tensors = fn_block(survival_rate=survival_rate)

    def train():
        return tf.cond(survive, lambda: survived_tensors, lambda: tensors)

    def test():
        return test_tensors

    return tf.cond(training, train, test)


def build_model(images, strokes, lengths, labels, training):
    """
    """
    activation = tf.nn.relu

    normalization = functools.partial(
        tf.contrib.layers.batch_norm, is_training=training)

    initializer = tf.truncated_normal_initializer(stddev=0.02)

    resnets_block = functools.partial(
        stochastic_block,
        stride=1,
        expansion=1,
        training=training,
        activation=activation,
        normalization=normalization,
        initializer=initializer)

    pooling_block = functools.partial(
        stochastic_block,
        stride=2,
        expansion=1,
        training=training,
        survival_rate=1.0,
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

    tensors = resnets_block(tensors, scope_name='r00', survival_rate=1.0)
#   tensors = resnets_block(tensors, scope_name='r01', survival_rate=0.9)
#   tensors = resnets_block(tensors, scope_name='r02', survival_rate=0.8)
#   tensors = resnets_block(tensors, scope_name='r03', survival_rate=0.7)
#   tensors = resnets_block(tensors, scope_name='r04', survival_rate=0.6)
#   tensors = resnets_block(tensors, scope_name='r05', survival_rate=0.5)

    tensors = pooling_block(tensors, scope_name='p10')
    tensors = resnets_block(tensors, scope_name='r10', survival_rate=1.0)
#   tensors = resnets_block(tensors, scope_name='r11', survival_rate=0.9)
#   tensors = resnets_block(tensors, scope_name='r12', survival_rate=0.8)
#   tensors = resnets_block(tensors, scope_name='r13', survival_rate=0.7)
#   tensors = resnets_block(tensors, scope_name='r14', survival_rate=0.6)
#   tensors = resnets_block(tensors, scope_name='r15', survival_rate=0.5)

    tensors = pooling_block(tensors, scope_name='p20')
    tensors = resnets_block(tensors, scope_name='r20', survival_rate=1.0)
#   tensors = resnets_block(tensors, scope_name='r21', survival_rate=0.9)
#   tensors = resnets_block(tensors, scope_name='r22', survival_rate=0.8)
#   tensors = resnets_block(tensors, scope_name='r23', survival_rate=0.7)
#   tensors = resnets_block(tensors, scope_name='r24', survival_rate=0.6)
#   tensors = resnets_block(tensors, scope_name='r25', survival_rate=0.5)

    tensors = pooling_block(tensors, scope_name='p30')
#   tensors = resnets_block(tensors, scope_name='r31', survival_rate=0.95)
    tensors = resnets_block(tensors, scope_name='r32', survival_rate=0.90)
#   tensors = resnets_block(tensors, scope_name='r33', survival_rate=0.85)
    tensors = resnets_block(tensors, scope_name='r34', survival_rate=0.80)
#   tensors = resnets_block(tensors, scope_name='r35', survival_rate=0.75)
    tensors = resnets_block(tensors, scope_name='r36', survival_rate=0.70)
#   tensors = resnets_block(tensors, scope_name='r37', survival_rate=0.65)
    tensors = resnets_block(tensors, scope_name='r38', survival_rate=0.60)
#   tensors = resnets_block(tensors, scope_name='r39', survival_rate=0.55)
    tensors = resnets_block(tensors, scope_name='r3a', survival_rate=0.50)

    tensors = tf.layers.conv2d(
        tensors,
        filters=1360,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=None,
        use_bias=False,
        kernel_initializer=initializer,
        name='block_final')

    tensors = normalization(tensors)

    tensors = activation(tensors)

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

