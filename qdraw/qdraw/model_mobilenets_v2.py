"""
"""
import tensorflow as tf


def block(
        tensors,
        expansion,
        stride,
        filters,
        training,
        scope_name):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    num_filters_in = tensors.shape[-1]

    source_tensors = tensors

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

        tensors = tf.layers.batch_normalization(tensors, training=training)

        tensors = tf.nn.relu(tensors)

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

        tensors = tf.layers.batch_normalization(tensors, training=training)

        tensors = tf.nn.relu(tensors)

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

        tensors = tf.layers.batch_normalization(tensors, training=training)

        if stride == 1 and num_filters_in == filters:
            tensors = tf.add(tensors, source_tensors)

    return tensors


def build_model(images, strokes, lengths, labels, training):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    tensors = images

    tensors = tf.layers.conv2d(
        tensors,
        filters=32,
        kernel_size=3,
        strides=2,
        padding='same',
        activation=None,
        use_bias=True,
        kernel_initializer=initializer,
        name='conv_in')

    tensors = tf.layers.batch_normalization(tensors, training=training)

    tensors = tf.nn.relu(tensors)

    # NOTE: expansion, stride, filters, training, name
    block_configs = [
        (1, 1, 16, training, 'block_0'),
        (6, 2, 24, training, 'block_1'),
        (6, 1, 24, training, 'block_2'),
        (6, 2, 32, training, 'block_3'),
        (6, 1, 32, training, 'block_4'),
        (6, 1, 32, training, 'block_5'),
        (6, 2, 64, training, 'block_6'),
        (6, 1, 64, training, 'block_7'),
        (6, 1, 64, training, 'block_8'),
        (6, 1, 64, training, 'block_9'),
        (6, 1, 96, training, 'block_10'),
        (6, 1, 96, training, 'block_11'),
        (6, 1, 96, training, 'block_12'),
        (6, 2, 160, training, 'block_13'),
        (6, 1, 160, training, 'block_14'),
        (6, 1, 160, training, 'block_15'),
        (6, 1, 320, training, 'block_16'),
    ]

    for params in block_configs:
        tensors = block(tensors, *params)

    tensors = tf.layers.conv2d(
        tensors,
        filters=1280,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=None,
        use_bias=False,
        kernel_initializer=initializer,
        name='block_final')

    tensors = tf.layers.batch_normalization(tensors, training=training)

    tensors = tf.nn.relu(tensors)

    tensors = \
        tf.nn.avg_pool(tensors, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

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

