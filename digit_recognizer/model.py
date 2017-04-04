"""
"""
import os
import tensorflow as tf

from six.moves import range


class Model(object):
    """
    """
    @staticmethod
    def network(source):
        """
        build the classifier network.
        param 'source' is the input data in shape [-1, 32, 32, 1].
        """
        weights_initializer = tf.truncated_normal_initializer(stddev=0.02)

        for layer_idx in range(5):

            # arXiv:1511.06434v2
            # in discriminator, use LeakyReLU
            source = tf.contrib.layers.convolution2d(
                inputs=source,
                num_outputs=2 ** (1 + layer_idx),
                kernel_size=[4, 4],
                stride=[2, 2],
                padding='SAME',
                activation_fn=tf.nn.relu,
                normalizer_fn=tf.contrib.layers.batch_norm,
                weights_initializer=weights_initializer,
                scope='conv_{}'.format(layer_idx))

        # for fully connected layer
        source = tf.contrib.layers.flatten(source)

        # fully connected layer to classify
        source = tf.contrib.layers.fully_connected(
            inputs=source,
            num_outputs=10,
            weights_initializer=weights_initializer,
            scope='out')

        return source

    def __init__(self):
        """
        """
        log_dir = tf.app.flags.FLAGS.log_dir
        checkpoint_dir = tf.app.flags.FLAGS.checkpoint_dir

        # sanity check: log_dir
        if log_dir is None or not os.path.isdir(log_dir):
            raise Exception('bad log_dir: {}'.format(log_dir))

        # sanity check: checkpoint_dir
        if checkpoint_dir is not None and not os.path.isdir(checkpoint_dir):
            raise Exception('bad checkpoint_dir: {}'.format(checkpoint_dir))

        global_step = tf.get_variable(
            "gstep",
            [],
            trainable=False,
            initializer=tf.constant_initializer(0.0))

        inputs = tf.placeholder(tf.float32, [None, 32, 32, 1])
        labels = tf.placeholder(tf.float32, [None, 10])

        logits = Model.network(inputs)
        result = tf.argmax(logits, 1)
        entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)

        loss = tf.reduce_sum(entropy)
        trainer = tf.train \
            .AdamOptimizer(1e-3) \
            .minimize(loss, global_step=global_step)

        self._loss = loss
        self._inputs = inputs
        self._labels = labels
        self._result = result
        self._trainer = trainer
        self._global_step = global_step

        self._checkpoint_source_path = \
            tf.train.latest_checkpoint(checkpoint_dir)
        self._checkpoint_target_path = \
            os.path.join(checkpoint_dir, 'model.ckpt')

        # build session
        self._session = tf.Session()

        # restore
        if self._checkpoint_source_path is None:
            self._session.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(
                self._session, self._checkpoint_source_path)

        self._summary_loss = tf.summary.scalar('loss', self._loss)

        # give up overlapped old data
        g_step = self._session.run(self._global_step)

        self._reporter = tf.summary.FileWriter(log_dir, self._session.graph)

        self._reporter.add_session_log(
            tf.SessionLog(status=tf.SessionLog.START),
            global_step=g_step)

    def predict(self, images):
        """
        """
        feeds = {self._inputs: images}

        return self._session.run(self._result, feeds)

    def save_checkpoint(self):
        """
        """
        saver = tf.train.Saver()

        saver.save(self._session, self._checkpoint_target_path,
                   global_step=self._global_step)

    def train(self, images, labels):
        """
        """
        fetch = [
            self._loss, self._global_step, self._summary_loss, self._trainer]

        feeds = {
            self._inputs: images,
            self._labels: labels,
        }

        loss, step, summary, _ = self._session.run(fetch, feeds)

        self._reporter.add_summary(summary, step)

        return loss, step
