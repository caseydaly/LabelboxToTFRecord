
import argparse
import os
from typing import List

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()


# Copies several tfrecord files into a single new one
def join_files(outfilename: str, infilenames: List[str]) -> None:
    
    total=0
    with tf.io.TFRecordWriter(outfilename) as writer: #type:ignore
        for infilename in infilenames:
            indataset = tf.data.TFRecordDataset(infilename) # type:ignore
            for rec_idx, rec in enumerate(indataset): #type: ignore
                writer.write(rec.numpy()) #type:ignore
                total+=1
            print(f"Finished writing {rec_idx+1} records to {outfilename}")

    print(f"Wrote a total of {total} records.")

    # TODO It would be nice if there was a better way of naming output files in
    # all of these scripts than just appending stuff to the end of the name
    # after the extension. See https://stackoverflow.com/a/45353565/356887
    os.rename(outfilename, outfilename + f"-{total}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine several .tfrecord files into a new one')
    parser.add_argument('outfile', metavar='outfile', type=str, nargs=1, help='the name of the output file')
    parser.add_argument('infiles', metavar='infiles', type=str, nargs='+', help='files to be combined')

    args = parser.parse_args()
    join_files(args.outfile[0], args.infiles)