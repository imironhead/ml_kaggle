"""
"""
import tensorflow as tf


def build_model(images, strokes, lengths, labels):
    """
    """
    tensors = strokes

    for index, (filters, kernel) in enumerate([(48, 5), (64, 5), (96, 3)]):
        if index > 0:
            tensors = tf.layers.dropout(
                tensors,
                rate=0.3,
                training=(labels is not None))

        tensors = tf.layers.conv1d(
            tensors,
            filters=filters,
            kernel_size=kernel,
            activation=None,
            strides=1,
            padding='same',
            name='conv1d_{}'.format(index))

    # NOTE: RNN
    # NOTE: Convolutions output [B, L, Ch], while CudnnLSTM is time-major.
    if True:
        tensors = tf.transpose(tensors, [1, 0, 2])

        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=3,
            num_units=128,
            dropout=0.0 if labels is None else 0.3,
            direction='bidirectional')

        tensors, states = lstm(tensors)

        # NOTE: Convert back from time-major outputs to batch-major outputs.
        tensors = tf.transpose(tensors, [1, 0, 2])
    else:
        cell = tf.nn.rnn_cell.BasicLSTMCell

        cells_fw = [cell(128) for _ in range(3)]
        cells_bw = [cell(128) for _ in range(3)]

#       cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
#       cells_bw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_bw]

        tensors, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=cells_fw,
            cells_bw=cells_bw,
            inputs=tensors,
            sequence_length=lengths,
            dtype=tf.float32,
            scope='rnn_classification')

    # NOTE: outputs is [batch_size, L, N] where L is the maximal sequence
    #       length and N the number of nodes in the last layer.
    mask = tf.sequence_mask(lengths, tf.shape(tensors)[1])
    mask = tf.expand_dims(mask, 2)
    mask = tf.tile(mask,[1, 1, tf.shape(tensors)[2]])

    tensors = tf.where(mask, tensors, tf.zeros_like(tensors))

    tensors = tf.reduce_sum(tensors, axis=1)

    logits = tf.layers.dense(tensors, 340)

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
