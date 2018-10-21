"""
"""
import tensorflow as tf


def block(tensors, filters, stride, training, scope_name):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    with tf.variable_scope(scope_name):
        # XXX: failed to replace slim!!!
        tensors = tf.contrib.slim.separable_convolution2d(tensors,
            num_outputs=None,
            stride=stride,
            depth_multiplier=1,
            kernel_size=[3, 3],
            scope='depthwise_conv')

        tensors = tf.contrib.slim.batch_norm(tensors, is_training=training)

        tensors = tf.layers.conv2d(
            tensors,
            filters=filters,
            kernel_size=1,
            strides=1,
            padding='same',
            data_format='channels_last',
            activation=tf.nn.relu,
            use_bias=False,
            kernel_initializer=initializer,
            name='conv_in')

        tensors = tf.contrib.slim.batch_norm(tensors, is_training=training)

    return tensors


def build_model(images, strokes, lengths, labels):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    training = tf.placeholder(shape=[], dtype=tf.bool)

    tensors = images

    tensors = tf.layers.conv2d(
        tensors,
        filters=64,
        kernel_size=3,
        strides=1,
        padding='same',
        data_format='channels_last',
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=initializer,
        name='conv_in')

    block_params = [
        (128, 2), (128, 1),
        (256, 2), (256, 1),
        (512, 2), (512, 1),
        (1024, 2), (1024, 1),
    ]

    for i, (filters, stride) in enumerate(block_params):
        tensors = block(
            tensors,
            filters,
            stride,
            training,
            'block_{}'.format(i))

    tensors = tf.nn.avg_pool(tensors, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

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

    # NOTE: build a simplier model without traning op
    if labels is None:
        return {
            'images': images,
            'strokes': strokes,
            'lengths': lengths,
            'logits': logits,
        }

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels,
        logits,
        reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)

    step = tf.train.get_or_create_global_step()

    learning_rate = tf.placeholder(shape=[], dtype=tf.float32)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
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
        'training': training,
        'optimizer': optimizer,
        'learning_rate': learning_rate,
    }

