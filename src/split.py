
import argparse
from typing import List, Iterator

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


# Splits tfrecord files into several files
def split_files(filename: str, splits: List[int]) -> None:
    dataset: tf.data.Dataset = tf.data.TFRecordDataset(filename)
    rec_counter: int = 0
    total_records: int = len([r for r in dataset])
    print(f"Found {total_records} records in source file.")
    if sum(splits) != total_records:
        raise ValueError(f"Sum of splits {sum(splits)} does not equal total number of records "
                         f"{total_records}")
    rec_iter:Iterator = iter(dataset)
    split: int
    for split_idx, split in enumerate(splits):
        outfile: str = filename + f".{split_idx}-{split}"
        with tf.io.TFRecordWriter(outfile) as writer:
            for out_idx in range(split):
                rec: tf.Tensor = next(rec_iter, None)
                rec_counter +=1
                writer.write(rec.numpy())
        print(f"Finished writing {split} records to file {split_idx}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split a .tfrecord file into smaller files')
    parser.add_argument('file', metavar='infile', type=str, nargs=1, help='the .tfrecord file to split')
    parser.add_argument('splits', metavar='splits', type=int, nargs='+', help='Space-separated '
        'list of integers, the number of records to put in each output file. Should add up to the '
        'total number of records in the input tfrecord file.'
    )

    args = parser.parse_args()
    split_files(args.file[0], args.splits)