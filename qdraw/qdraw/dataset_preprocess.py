"""
"""
import csv
import functools
import itertools
import json
import multiprocessing.dummy
import os
import random

import numpy as np
import skimage.draw
import skimage.io
import tensorflow as tf

import qdraw.dataset as dataset


def perturb_strokes(strokes):
    """
    """
    lt = np.random.randint(8, size=2)
    lb = np.random.randint(8, size=2) + np.array([0, 248])
    rt = np.random.randint(8, size=2) + np.array([248, 0])
    rb = np.random.randint(8, size=2) + np.array([248, 248])

    new_strokes = []

    for xs, ys in strokes:
        new_xs = []
        new_ys = []

        for x, y in zip(xs, ys):
            p = \
                lt * (256 - x) * (256 - y) + \
                lb * (256 - x) * y + \
                rt * (256 - y) * x + \
                rb * x * y

            p //= 65536

            new_xs.append(p[0])
            new_ys.append(p[1])

        new_strokes.append([new_xs, new_ys])

    return new_strokes


def center_strokes(strokes):
    """
    """
    def extremum(ls, idx, fun):
        """
        """
        for i, points in enumerate(ls):
            m = fun(points[idx])

            n = m if i == 0 else fun(n, m)

        return n

    min_x, min_y = extremum(strokes, 0, min), extremum(strokes, 1, min)
    max_x, max_y = extremum(strokes, 0, max), extremum(strokes, 1, max)

    dx = (256 - (max_x - min_x)) // 2 - min_x
    dy = (256 - (max_y - min_y)) // 2 - min_y

    output_strokes = []

    for xs, ys in strokes:
        xs = [x + dx for x in xs]
        ys = [y + dy for y in ys]

        output_strokes.append((xs, ys))

    return output_strokes


def strokes_to_series(strokes):
    """
    cordinates range from 0 to 255
    """
    num_points = sum([len(xs) for xs, ys in strokes])

    points = np.zeros((num_points, 3), dtype=np.uint8)

    base = 0

    for xs, ys in strokes:
        for i in range(len(xs)):
            points[base + i, 0] = xs[i]
            points[base + i, 1] = ys[i]

        base += len(xs)

        points[base - 1, 2] = 255

    return points.flatten().tostring()


def strokes_to_image(strokes, image_size):
    """
    import cv2

    image = np.zeros((256, 256), dtype=np.uint8)

    for t, (xs, ys) in enumerate(strokes):
        color = 255 - min(t, 10) * 13

        for i in range(1, len(xs)):
            cv2.line(image, (xs[i-1], ys[i-1]), (xs[i], ys[i]), color, 6)

    if image_size != 256:
        image = cv2.resize(image, (image_size, image_size))
    """
    scale = 256 // image_size

    image = np.zeros((image_size, image_size), dtype=np.uint8)

    for t, (xs, ys) in enumerate(strokes):
        color = 255 - min(t, 10) * 13

        for i in range(1, len(xs)):
            rr, cc, val = skimage.draw.line_aa(
                ys[i-1] // scale,
                xs[i-1] // scale,
                ys[i] // scale,
                xs[i] // scale)

            image[rr, cc] = val * color

    return image.flatten().tostring()


def int64_feature(v):
    """
    create a feature which contains a 64-bits integer
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))


def raw_feature(v):
    """
    create a feature which contains 32-bits floats in binary format.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def row_to_example_training(row, image_size, perturb):
    """
    """
    # NOTE: strokes
    strokes = json.loads(row[1])

    # NOTE: key id
    keyid = int(row[2])

    # NOTE: recognized
    recognized = 1 if row[3].lower() == 'true' else 0

    # NOTE: skip timestamp

    # NOTE: label, replace space with _ for kaggle quick draw competition
    label = dataset.label_to_index[row[5].replace(' ', '_')]

    # NOTE:
    if perturb:
        strokes = perturb_strokes(strokes)

    # NOTE: process strokes
    strokes = center_strokes(strokes)

    series = strokes_to_series(strokes)

    image = strokes_to_image(strokes, image_size)

    # NOTE: make example
    feature = {
        'keyid': int64_feature(keyid),
        'strokes': raw_feature(series),
        'image': raw_feature(image),
        'recognized': int64_feature(recognized),
        'label': int64_feature(label),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def row_to_example_testing(row, image_size, perturb):
    """
    """
    # NOTE: key id
    keyid = int(row[0])

    # NOTE: strokes
    strokes = json.loads(row[2])

    # NOTE:
    if perturb:
        strokes = perturb_strokes(strokes)

    # NOTE: process strokes
    strokes = center_strokes(strokes)

    series = strokes_to_series(strokes)

    image = strokes_to_image(strokes, image_size)

    # NOTE: make example
    feature = {
        'keyid': int64_feature(keyid),
        'strokes': raw_feature(series),
        'image': raw_feature(image),
        'recognized': int64_feature(-1),
        'label': int64_feature(-1),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def row_to_example(row, image_size, perturb):
    """
    """
    if len(row) == 6:
        return row_to_example_training(row, image_size, perturb)
    elif len(row) == 3:
        return row_to_example_testing(row, image_size, perturb)


def preprocess(description):
    """
    """
    fn_row_to_example = functools.partial(
        row_to_example,
        image_size=description['image_size'],
        perturb=description['perturb'])

    with open(description['source_path'], newline='') as csv_file:
        rows = csv.reader(csv_file, delimiter=',')

        return [fn_row_to_example(row) for row in rows]


def collect_source_path_generators():
    """
    assume csv with the same label reside in the same directory
    """
    def path_generator(paths):
        """
        """
        while True:
            random.shuffle(paths)
            for path in paths:
                yield path

    FLAGS = tf.app.flags.FLAGS

    source_path_generators = []

    for dir_path, dir_names, file_names in os.walk(FLAGS.source_dir):
        if len(file_names) == 0:
            continue

        # NOTE: only csv which is raw data
        source_names = [n for n in file_names if n.endswith('.csv')]
        source_paths = [os.path.join(dir_path, n) for n in source_names]

        source_path_generators.append(path_generator(source_paths))

    return source_path_generators


def preprocess_training():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    source_path_generators = collect_source_path_generators()

    with multiprocessing.dummy.Pool(8) as pool:
        for i in range(FLAGS.num_output):
            descriptions = []

            # NOTE: collect csv from different categories
            for path_generator in source_path_generators:
                descriptions.append({
                    'source_path': next(path_generator),
                    'image_size': FLAGS.image_size,
                    'perturb': FLAGS.perturb})

            example_groups = pool.map(preprocess, descriptions)

            examples = list(itertools.chain.from_iterable(example_groups))

            # NOTE: no shuffling for testing set
            if FLAGS.shuffle:
                random.shuffle(examples)

            # NOTE: build result name with prefix and index
            result_name = '{}_{:0>4}.tfrecord.gz'.format(FLAGS.prefix, i)
            result_path = os.path.join(FLAGS.result_dir, result_name)

            # NOTE: write gzip
            options = tf.python_io.TFRecordOptions(
                tf.python_io.TFRecordCompressionType.GZIP)

            with tf.python_io.TFRecordWriter(result_path, options=options) as writer:
                for example in examples:
                    writer.write(example.SerializeToString())

            print('done: {}'.format(result_path))


def preprocess_testing():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    description = {
        'source_path': FLAGS.source_csv_path,
        'image_size': FLAGS.image_size,
        'perturb': FLAGS.perturb,
    }

    examples = preprocess(description)

    # NOTE: write gzip
    options = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.GZIP)

    with tf.python_io.TFRecordWriter(FLAGS.result_tfr_path, options=options) as writer:
        for example in examples:
            writer.write(example.SerializeToString())

    print('done: {}'.format(FLAGS.result_tfr_path))


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    if FLAGS.source_dir and tf.gfile.Exists(FLAGS.source_dir):
        preprocess_training()

    if FLAGS.source_csv_path and tf.gfile.Exists(FLAGS.source_csv_path):
        preprocess_testing()


if __name__ == '__main__':
    # NOTE: to handle single csv (test.csv, no shuffle)
    tf.app.flags.DEFINE_string('source_csv_path', None, '')
    tf.app.flags.DEFINE_string('result_tfr_path', None, '')

    tf.app.flags.DEFINE_string('source_dir', None, '')
    tf.app.flags.DEFINE_string('result_dir', None, '')

    tf.app.flags.DEFINE_string('prefix', '', '')

    tf.app.flags.DEFINE_boolean('perturb', False, '')
    tf.app.flags.DEFINE_boolean('shuffle', False, '')

    tf.app.flags.DEFINE_integer('image_size', 32, '')
    tf.app.flags.DEFINE_integer('num_output', 64, '')

    tf.app.run()

