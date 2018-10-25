"""
"""
import tensorflow as tf


def bottleneck(
        tensors,
        num_filters_in,
        num_filters_out,
        expansion,
        stride,
        training,
        scope_name):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    source_tensors = tensors

    with tf.variable_scope(scope_name):
        tensors = tf.layers.conv2d(
            tensors,
            filters=num_filters_in * expansion,
            kernel_size=1,
            strides=1,
            padding='same',
            data_format='channels_last',
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=initializer,
            name='head')

        tensors = tf.contrib.slim.batch_norm(tensors, is_training=training)

        tensors = tf.contrib.slim.separable_convolution2d(
            tensors,
            num_outputs=None,
            stride=stride,
            depth_multiplier=1,
            kernel_size=[3, 3],
            scope='depthwise')

        tensors = tf.contrib.slim.batch_norm(tensors, is_training=training)

        tensors = tf.layers.conv2d(
            tensors,
            filters=num_filters_out,
            kernel_size=1,
            strides=1,
            padding='same',
            data_format='channels_last',
            activation=None,
            use_bias=True,
            kernel_initializer=initializer,
            name='tail')

        if stride == 1:
            if num_filters_in != num_filters_out:
                source_tensors = tf.layers.conv2d(
                    source_tensors,
                    filters=num_filters_out,
                    kernel_size=1,
                    strides=1,
                    padding='same',
                    data_format='channels_last',
                    activation=None,
                    use_bias=False,
                    kernel_initializer=initializer,
                    name='scale')

            tensors = tf.add(tensors, source_tensors)

    return tensors


def bottlenecks(
        tensors,
        num_filters_in,
        num_filters_out,
        expansion,
        stride,
        repeat,
        training,
        scope_name):
    """
    """
    with tf.variable_scope(scope_name):
        tensors = bottleneck(
            tensors,
            num_filters_in,
            num_filters_out,
            expansion,
            stride,
            training,
            'block_0')

        for index in range(1, repeat):
            tensors = bottleneck(
                tensors,
                num_filters_in,
                num_filters_out,
                expansion,
                1,
                training,
                'block_{}'.format(index))

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

    # NOTE: num_filters_in, num_filters_out, expansion, stride, repeat, name
    block_configs = [
#       (64, 64, 1, 1, 2, training, 'bottleneck_0'),
        (64, 128, 1, 2, 2, training, 'bottleneck_1'),
#       (128, 128, 1, 1, 4, training, 'bottleneck_2'),
        (128, 256, 1, 2, 4, training, 'bottleneck_3'),
#       (256, 256, 1, 1, 4, training, 'bottleneck_4'),
        (256, 512, 1, 2, 4, training, 'bottleneck_5'),
#       (512, 512, 1, 1, 4, training, 'bottleneck_6'),
        (512, 1024, 1, 2, 4, training, 'bottleneck_7'),
    ]

    for params in block_configs:
        tensors = bottlenecks(tensors, *params)

    tensors = tf.layers.conv2d(
        tensors,
        filters=2048,
        kernel_size=1,
        strides=1,
        padding='same',
        data_format='channels_last',
        activation=tf.nn.relu,
        use_bias=True,
        kernel_initializer=initializer,
        name='bottleneck_final')

    tensors = tf.nn.avg_pool(tensors, [1, 4, 4, 1], [1, 4, 4, 1], padding='SAME')

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

