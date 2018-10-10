"""
"""
import argparse
import csv
import gzip

import numpy as np

import qdraw.dataset as dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='finalize the result')

    parser.add_argument('--source_npz_path', type=str)
    parser.add_argument('--source_csv_path', type=str)
    parser.add_argument('--result_zip_path', type=str)

    args = parser.parse_args()

    logits = np.load(args.source_npz_path)

    logits = logits['logits']

    ranks = np.argsort(logits, axis=1)

    ranks = ranks[:, -1:-4:-1]

    # NOTE: read ids
    with open(args.source_csv_path, newline='') as csv_file:
        draws = csv.reader(csv_file, delimiter=',')

        # NOTE: skip header
        next(draws)

        ids = [draw[0] for draw in draws]

    # NOTE:
    with gzip.open(args.result_zip_path, mode='wt', newline='') as zip_file:
        writer = csv.writer(zip_file, delimiter=',')

        writer.writerow(['key_id', 'word'])

        for idx, _id in enumerate(ids):
            labels = [dataset.index_to_label[ranks[idx, i]] for i in range(3)]

            labels = ' '.join(labels)

            writer.writerow([_id, labels])

