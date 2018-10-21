"""
"""
import tensorflow as tf


def self_attention(tensors, num_out_channels, num_aux_channels, scope_name):
    """
    """
    shape = tf.shape(tensors)

    batch_size, height, width = [shape[i] for i in range(3)]

    initializer = tf.truncated_normal_initializer(stddev=0.02)

    with tf.variable_scope(scope_name):
        fx = tf.layers.conv2d(
            tensors,
            filters=num_aux_channels,
            kernel_size=1,
            strides=1,
            padding='same',
            data_format='channels_last',
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=initializer,
            name='fx')

        gx = tf.layers.conv2d(
            tensors,
            filters=num_aux_channels,
            kernel_size=1,
            strides=1,
            padding='same',
            data_format='channels_last',
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=initializer,
            name='gx')

        hx = tf.layers.conv2d(
            tensors,
            filters=num_out_channels,
            kernel_size=1,
            strides=1,
            padding='same',
            data_format='channels_last',
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=initializer,
            name='hx')

        gamma = tf.get_variable(
            'gamma',
            [],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        fx = tf.reshape(fx, [batch_size, width * height, num_aux_channels])
        gx = tf.reshape(gx, [batch_size, width * height, num_aux_channels])
        hx = tf.reshape(hx, [batch_size, width * height, num_out_channels])
        mx = tf.matmul(fx, gx, transpose_b=True)
        mx = tf.nn.softmax(mx, axis=2)
        o = tf.matmul(mx, hx)
        o = tf.reshape(o, [batch_size, height, width, num_out_channels])

        tensors = gamma * o + tensors

    return tensors


def build_model(images, strokes, lengths, labels):
    """
    images: [None, size, size, 1] float32
    labels: [None], 340 categories
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    tensors = images

    tensors = tf.layers.conv2d(
        tensors,
        filters=32,
        kernel_size=3,
        strides=1,
        padding='same',
        data_format='channels_last',
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=initializer,
        name='conv_in')

    tensors = self_attention(tensors, 32, 4, 'conv_in_attention')

    for index, filters in enumerate([64, 128, 256]):
        tensors = tf.layers.conv2d(
            tensors,
            filters=filters,
            kernel_size=3,
            strides=2,
            padding='same',
            data_format='channels_last',
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=initializer,
            name='conv_{}'.format(index))

        tensors = self_attention(
            tensors, filters, filters // 8, 'conv_{}_attention'.format(index))

    embeddings = tensors

    # NOTE: flatten for fc
    tensors = tf.layers.flatten(embeddings)

    # NOTE
    for i, units in enumerate([2048, 1024]):
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
        use_bias=True,
        kernel_initializer=initializer,
        name='final')

    logits = tf.identity(logits, name='logits')

    # NOTE: build a simplier model without traning op
    if labels is None:
        return {
            'images': images,
            'strokes': strokes,
            'lengths': lengths,
            'logits': logits,
        }

    loss_classification = tf.losses.sparse_softmax_cross_entropy(
        labels,
        logits,
        reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

    tensors = embeddings

    for index, filters in enumerate([128, 64, 32]):
        tensors = tf.layers.conv2d_transpose(
            tensors,
            filters=filters,
            kernel_size=3,
            strides=2,
            padding='same',
            data_format='channels_last',
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=initializer,
            name='dconv_{}'.format(index))

    tensors = tf.layers.conv2d(
        tensors,
        filters=1,
        kernel_size=3,
        strides=1,
        padding='same',
        data_format='channels_last',
        activation=tf.nn.tanh,
        use_bias=True,
        kernel_initializer=initializer,
        name='conv_out')

    loss_reconstruction = tf.losses.mean_squared_error(
        images, tensors, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

    loss = loss_classification + loss_reconstruction

    step = tf.train.get_or_create_global_step()

    learning_rate = tf.placeholder(shape=[], dtype=tf.float32)

    optimizer = tf.train \
        .AdamOptimizer(learning_rate=learning_rate) \
        .minimize(loss, global_step=step)

    return {
        'images': images,
        'labels': labels,
        'strokes': strokes,
        'lengths': lengths,
        'logits': logits,
        'loss': loss,
        'step': step,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
    }

