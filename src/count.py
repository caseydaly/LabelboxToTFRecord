
import argparse
import os
from typing import List

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


def count(infilenames: List[str], displaytotal: bool=False) -> None:
    
    total: int = 0
    subtotal: int = 0
    for infilename in infilenames:
        indataset = tf.data.TFRecordDataset(infilename) # type:ignore
        subtotal = len([r for r in indataset]) # type: ignore
        total+=subtotal
        if not displaytotal:
            print(subtotal)

    if displaytotal:
        print(total)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display the number of records in each file')
    parser.add_argument('infiles', metavar='infiles', type=str, nargs='+', help='files with records to be counted')
    parser.add_argument('--total', action="store_true", help='instead of the the total for each file, display the sum total across all files')
    # TODO an arg for displaying info about the category breakdown for that file (how many sharks, how many people, etc.)

    args = parser.parse_args()
    count(args.infiles, args.total)