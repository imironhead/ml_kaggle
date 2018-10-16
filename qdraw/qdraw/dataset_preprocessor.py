"""
https://stackoverflow.com/questions/42394585/how-to-inspect-a-tensorflow-tfrecord-file

for example in tf.python_io.tf_record_iterator("data/foobar.tfrecord"):
    result = tf.train.Example.FromString(example)
"""
import csv
import functools
import json
import multiprocessing.dummy
import os
import random

import numpy as np
import skimage.draw
import skimage.io
import tensorflow as tf

import qdraw.dataset as dataset


def perturb_strokes(strokes, d):
    """
    """
    output_strokes = []

    for xs, ys in strokess:
        xs = [x + random.randint(-d, d) for x in xs]
        ys = [y + random.randint(-d, d) for y in ys]

        output_strokes.append((xs, ys))

    # NOTE: rotate

    return output_strokes


def normalize_strokes_to_uniform(strokes):
    """
    """
    def extremum(ls, idx, fun):
        """
        """
        for i, points in enumerate(ls):
            m = fun(points[idx])

            n = m if i == 0 else fun(n, m)

        return float(n)

    output_strokes = []

    min_x, min_y = extremum(strokes, 0, min), extremum(strokes, 1, min)
    max_x, max_y = extremum(strokes, 0, max), extremum(strokes, 1, max)
    mid_x, mid_y = 0.5 * (max_x + min_x), 0.5 * (max_y + min_y)

    # NOTE: scale
    s = 2.0 / max(max_x - min_x, max_y - min_y)

    for xs, ys in strokes:
        xs = [(x - mid_x) * s for x in xs]
        ys = [(y - mid_y) * s for y in ys]

        output_strokes.append((xs, ys))

    return output_strokes


def normalize_strokes_to_image(strokes, image_size):
    """
    """
    def extremum(ls, idx, fun):
        """
        """
        for i, points in enumerate(ls):
            m = fun(points[idx])

            n = m if i == 0 else fun(n, m)

        return n

    output_strokes = []

    min_x, min_y = extremum(strokes, 0, min), extremum(strokes, 1, min)
    max_x, max_y = extremum(strokes, 0, max), extremum(strokes, 1, max)

    # NOTE: scale to fix image_size
    s = max(max_x - min_x, max_y - min_y)
    t = image_size - 1

    for xs, ys in strokes:
        xs = [(x - min_x) * t // s for x in xs]
        ys = [(y - min_y) * t // s for y in ys]

        output_strokes.append((xs, ys))

    strokes, output_strokes = output_strokes, []

    # NOTE: move to center
    tx = (t - extremum(strokes, 0, max)) // 2
    ty = (t - extremum(strokes, 1, max)) // 2

    for xs, ys in strokes:
        xs = [x + tx for x in xs]
        ys = [y + ty for y in ys]

        output_strokes.append((xs, ys))

    return output_strokes


def strokes_to_image(strokes, image_size):
    """
    """
    image = np.zeros((image_size, image_size), dtype=np.uint8)

    for xs, ys in strokes:
        for i in range(1, len(xs)):
            rr, cc = skimage.draw.line(ys[i-1], xs[i-1], ys[i], xs[i])

            image[rr, cc] = 255

    return image


def strokes_to_points(strokes):
    """
    """
    num_points = sum([len(xs) for xs, ys in strokes])

    points = np.zeros((num_points, 3), dtype=np.float32)

    base = 0

    for xs, ys in strokes:
        for i in range(len(xs)):
            points[base + i, 0] = xs[i]
            points[base + i, 1] = ys[i]

        base += len(xs)

        points[base - 1, 2] = 1.0

    return points.flatten().tostring()


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


def draw_to_complex_example(draw, image_size, add_perturbation):
    """
    """
    strokes = draw[1]

    if add_perturbation:
        pass

    strokes_uniform = normalize_strokes_to_uniform(strokes)

    strokes_image = normalize_strokes_to_image(strokes, image_size)

    points = strokes_to_points(strokes_uniform)

    image = strokes_to_image(strokes_image, image_size)

    feature = {}

    feature['keyid'] = int64_feature(draw[0])

    feature['strokes'] = \
        tf.train.Feature(bytes_list=tf.train.BytesList(value=[points]))

    feature['image'] = image_feature(image)

    if len(draw) > 2:
        feature['label'] = int64_feature(draw[2])

    return tf.train.Example(features=tf.train.Features(feature=feature))


def columns_to_strokes_and_label(
        columns, index_keyid, index_strokes, index_label):
    """
    """
    keyid = int(columns[index_keyid])

    strokes = json.loads(columns[index_strokes])

    if index_label >= 0:
        label = columns[index_label].replace(' ', '_')
        label = dataset.label_to_index[label]

        return (keyid, strokes, label)
    else:
        return (keyid, strokes,)


def preprocess(description):
    """
    """
    columns_to_draw = functools.partial(
        columns_to_strokes_and_label,
        index_keyid=description['index_keyid'],
        index_strokes=description['index_strokes'],
        index_label=description['index_label'])

    draws = []

    for source_path in description['source_paths']:
        with open(source_path, newline='') as csv_file:
            csv_draws = csv.reader(csv_file, delimiter=',')

            draws.extend([columns_to_draw(columns) for columns in csv_draws])

    random.shuffle(draws)

    draw_to_example = functools.partial(
        draw_to_complex_example,
        image_size=description['image_size'],
        add_perturbation=description['add_perturbation'])

    # NOTE: load images
    with tf.python_io.TFRecordWriter(description['result_path']) as writer:
        for draw in draws:
            example = draw_to_example(draw)

            writer.write(example.SerializeToString())

    print('done: {}'.format(description['result_path']))


def collect_source_paths():
    """
    assume csv with the same label reside in the same directory
    """
    FLAGS = tf.app.flags.FLAGS

    source_paths_collection = []

    for dir_path, dir_names, file_names in os.walk(FLAGS.source_dir):
        if len(file_names) == 0:
            continue

        # NOTE: only csv which is raw data
        source_names = [n for n in file_names if n.endswith('.csv')]
        source_paths = [os.path.join(dir_path, n) for n in source_names]

        source_paths_collection.append(source_paths)

    random.shuffle(source_paths_collection)

    return source_paths_collection


def preprocess_training():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    source_paths_collection = collect_source_paths()

    # NOTE: currently, instead of reusing data, we dorp remainder here
    num_output = min([len(ps) for ps in source_paths_collection])

    with multiprocessing.dummy.Pool(128) as pool:
        for i in range(0, num_output, 128):
            descriptions = []

            for index in range(i, min(i + 128, num_output)):
                # NOTE: build result name with prefix and index
                result_name = '{}_{:0>4}.tfrecord'.format(FLAGS.prefix, index)
                result_path = os.path.join(FLAGS.result_dir, result_name)

                # NOTE: mix source from each categories
                source_paths = [ps[index] for ps in source_paths_collection]

                # NOTE: build args for one task
                description = {
                    'source_paths': source_paths,
                    'result_path': result_path,
                    'image_size': FLAGS.image_size,
                    'index_keyid': FLAGS.index_keyid,
                    'index_label': FLAGS.index_label,
                    'index_strokes': FLAGS.index_strokes,
                    'add_perturbation': FLAGS.add_perturbation,
                }

                descriptions.append(description)

            pool.map(preprocess, descriptions)


def preprocess_testing():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    columns_to_draw = functools.partial(
        columns_to_strokes_and_label,
        index_keyid=FLAGS.index_keyid,
        index_strokes=FLAGS.index_strokes,
        index_label=FLAGS.index_label)

    with open(FLAGS.source_csv_path, newline='') as csv_file:
        csv_draws = csv.reader(csv_file, delimiter=',')

        draws = [columns_to_draw(columns) for columns in csv_draws]

    draw_to_example = functools.partial(
        draw_to_complex_example,
        image_size=FLAGS.image_size,
        add_perturbation=FLAGS.add_perturbation)

    # NOTE: load images
    with tf.python_io.TFRecordWriter(FLAGS.result_tfr_path) as writer:
        for draw in draws:
            example = draw_to_example(draw)

            writer.write(example.SerializeToString())


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    if FLAGS.source_dir is not None and tf.gfile.Exists(FLAGS.source_dir):
        preprocess_training()

    if FLAGS.source_csv_path is not None and tf.gfile.Exists(FLAGS.source_csv_path):
        preprocess_testing()


if __name__ == '__main__':
    # NOTE: to handle single csv (test.csv, no shuffle)
    tf.app.flags.DEFINE_string('source_csv_path', None, '')
    tf.app.flags.DEFINE_string('result_tfr_path', None, '')

    tf.app.flags.DEFINE_string('source_dir', None, '')
    tf.app.flags.DEFINE_string('result_dir', None, '')
    tf.app.flags.DEFINE_string('prefix', '', '')

    tf.app.flags.DEFINE_boolean('add_perturbation', False, '')

    tf.app.flags.DEFINE_integer('image_size', 32, '')

    tf.app.flags.DEFINE_integer('index_keyid', -1, '')
    tf.app.flags.DEFINE_integer('index_label', 5, '')
    tf.app.flags.DEFINE_integer('index_strokes', 1, '')

    tf.app.run()

