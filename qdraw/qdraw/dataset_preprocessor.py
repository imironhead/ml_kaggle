"""
"""
import csv
import json
import math
import multiprocessing.dummy
import os
import random

import numpy as np
import skimage.draw
import skimage.io
import tensorflow as tf

import qdraw.dataset as dataset


def perturb(lines, d):
    """
    """
    output_lines = []

    for xs, ys in lines:
        xs = [x + random.randint(-d, d) for x in xs]
        ys = [y + random.randint(-d, d) for y in ys]

        output_lines.append((xs, ys))

    # NOTE: rotate

    return output_lines


def normalize(lines, image_size):
    """
    """
    def extremum(ls, idx, fun):
        """
        """
        for i, points in enumerate(ls):
            m = fun(points[idx])

            n = m if i == 0 else fun(n, m)

        return n

    output_lines = []

    min_x, min_y = extremum(lines, 0, min), extremum(lines, 1, min)
    max_x, max_y = extremum(lines, 0, max), extremum(lines, 1, max)

    # NOTE: scale to fix image_size
    s = max(max_x - min_x, max_y - min_y)
    t = image_size - 1

    for xs, ys in lines:
        xs = [(x - min_x) * t // s for x in xs]
        ys = [(y - min_y) * t // s for y in ys]

        output_lines.append((xs, ys))

    lines, output_lines = output_lines, []

    # NOTE: move to center
    tx = (t - extremum(lines, 0, max)) // 2
    ty = (t - extremum(lines, 1, max)) // 2

    for xs, ys in lines:
        xs = [x + tx for x in xs]
        ys = [y + ty for y in ys]

        output_lines.append((xs, ys))

    return output_lines


def lines_to_image(image_size, lines, add_perturbation=False):
    """
    """
    if perturb:
        lines = perturb(lines, 4)

    lines = normalize(lines, image_size)

    image = np.zeros((image_size, image_size), dtype=np.uint8)

    for xs, ys in lines:
        for i in range(1, len(xs)):
            rr, cc = skimage.draw.line(ys[i-1], xs[i-1], ys[i], xs[i])

            image[rr, cc] = 255

    return image


def int64_feature(v):
    """
    create a feature which contains a 64-bits integer
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))


def image_feature(image):
    """
    create a feature which contains 32-bits floats in binary format.
    """
    image = image.astype(np.uint8).tostring()

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))


def preprocess_train():
    """
    """


def preprocess_test():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    preprocessed_data = []

    # NOTE: preprocess all image
    with open(FLAGS.source_csv, newline='') as csv_file:
        draws = csv.reader(csv_file, delimiter=',')

        # NOTE: skip header
        next(draws)

        for draw in draws:
            lines = json.loads(draw[2])
            image = lines_to_image(
                FLAGS.image_size, lines, FLAGS.block_begin > 0)

            preprocessed_data.append(image)

    # NOTE: output to 2 tfrecord
    with tf.python_io.TFRecordWriter(FLAGS.result_tfr) as writer:
        for image in preprocessed_data:
            feature = {'image': image_feature(image)}

            example = tf.train.Example(
                features=tf.train.Features(feature=feature))

            writer.write(example.SerializeToString())


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    if FLAGS.source_dir is not None and tf.gfile.Exists(FLAGS.source_dir):
        preprocess_train()

    if FLAGS.source_csv is not None and tf.gfile.Exists(FLAGS.source_csv):
        preprocess_test()


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('source_dir', None, '')
    tf.app.flags.DEFINE_string('result_dir', None, '')

    tf.app.flags.DEFINE_string('source_csv', None, '')
    tf.app.flags.DEFINE_string('result_tfr', None, '')

    tf.app.flags.DEFINE_integer('block_begin', 0, '')
    tf.app.flags.DEFINE_integer('block_end', 1, '')

    tf.app.flags.DEFINE_integer('image_size', 28, '')

    tf.app.run()

