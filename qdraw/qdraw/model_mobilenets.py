"""
"""
import tensorflow as tf


def block(tensors, filters, stride, channel_multiplier, training, scope_name):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    with tf.variable_scope(scope_name):
        in_channels = tensors.shape[-1]

        filter_weights = tf.get_variable(
            'depthwise_conv_filter',
            [3, 3, in_channels, channel_multiplier],
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
            name='pointwise_conv')

        tensors = tf.layers.batch_normalization(tensors, training=training)

        tensors = tf.nn.relu(tensors)

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
        use_bias=False,
        kernel_initializer=initializer,
        name='conv_in')

    tensors = tf.layers.batch_normalization(tensors, training=training)

    tensors = tf.nn.relu(tensors)

    block_params = [
        (64, 2), (64, 1),
        (128, 2), (128, 1),
        (256, 2), (256, 1),
        (512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
        (1024, 2), (1024, 1),
    ]

    for i, (filters, stride) in enumerate(block_params):
        tensors = block(
            tensors,
            filters,
            stride,
            1,
            training,
            'block_{}'.format(i))

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

    logits = tf.identity(logits, name='logits')

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

