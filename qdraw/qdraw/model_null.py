"""
"""
import tensorflow as tf


def build_model(images, strokes, lengths, labels, training):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    tensors = images

    tensors = tf.layers.conv2d(
        tensors,
        filters=1,
        kernel_size=3,
        strides=2,
        padding='same',
        activation=None,
        use_bias=False,
        kernel_initializer=initializer,
        name='conv_in')

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

