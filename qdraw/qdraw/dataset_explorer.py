"""
"""
import argparse
import csv
import json
import multiprocessing
import os

import numpy as np
import skimage.draw
import skimage.io


def report_image_size(csv_path):
    """
    """
    min_x, max_x = +1000, -1000
    min_y, max_y = +1000, -1000

    with open(csv_path, newline='') as csv_file:
        draws = csv.reader(csv_file, delimiter=',')

        # NOTE: skip header
        next(draws)

        for draw in draws:
            lines = json.loads(draw[1])

            for line in lines:
                min_x = min(min_x, min(line[0]))
                min_y = min(min_y, min(line[1]))
                max_x = max(max_x, max(line[0]))
                max_y = max(max_y, max(line[1]))

    print('done: {}'.format(csv_path))

    return min_x, min_y, max_x, max_y


def report_images_size(args):
    """
    """
    min_xs, min_ys, max_xs, max_ys = [], [], [], []

    csv_names = os.listdir(args.source_dir)

    csv_paths = [os.path.join(args.source_dir, n) for n in csv_names]

    with multiprocessing.Pool(16) as pool:
        xys = pool.imap_unordered(report_image_size, csv_paths, 20)

        for xy in xys:
            min_xs.append(xy[0])
            min_ys.append(xy[1])
            max_xs.append(xy[2])
            max_ys.append(xy[3])

    print('left:   {}'.format(min(min_xs)))
    print('top:    {}'.format(min(min_ys)))
    print('right:  {}'.format(max(max_xs)))
    print('bottom: {}'.format(max(max_ys)))


def dump_image(args):
    """
    """
    source_csv_path, result_dir_path = args

    basename = os.path.splitext(source_csv_path)[0]
    basename = os.path.basename(basename)
    basename = basename.replace(' ', '_')

    with open(source_csv_path, newline='') as csv_file:
        draws = csv.reader(csv_file, delimiter=',')

        # NOTE: skip header
        next(draws)

        image = None

        for j, draw in enumerate(draws):
            if image is None:
                image = np.zeros((1024, 1024), dtype=np.uint8)

            bx, by = (j % 1024) % 32 * 32, (j % 1024) // 32 * 32

            lines = json.loads(draw[1])

            for xs, ys in lines:
                xs = [bx + x * (32 - 1) // 255 for x in xs]
                ys = [by + y * (32 - 1) // 255 for y in ys]

                for i in range(1, len(xs)):
                    rr, cc = skimage.draw.line(ys[i-1], xs[i-1], ys[i], xs[i])

                    image[rr, cc] = 255

            if j % 1024 == 1023:
                result_path = os.path.join(
                    result_dir_path, '{}_{}.png'.format(basename, j - 1023))

                skimage.io.imsave(result_path, image)

                image = None

                print('saved: {}'.format(result_path))

        if image is not None:
            result_path = os.path.join(
                result_dir_path, '{}_fff.png'.format(basename))

            skimage.io.imsave(result_path, image)


def dump_images(args):
    """
    """
    csv_names = os.listdir(args.source_dir)

    csv_paths = [os.path.join(args.source_dir, n) for n in csv_names]

    new_args = [(p, args.result_dir) for p in csv_paths]

    with multiprocessing.Pool(16) as pool:
        pool.map(dump_image, new_args, 20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='explore quick draw dataset')

    parser.add_argument('--source_dir', type=str)
    parser.add_argument('--result_dir', type=str)

    args = parser.parse_args()

    # NOTE: number of training images: 49707919
    #       ls | xargs -I % wc -l % | awk '{ c += $1} END { print c}'

    # NOTE: simplified
    #       0, 0, 255, 255
    #report_images_size(args)

    dump_images(args)

