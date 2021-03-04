
import argparse
import os
from typing import List, Dict, Set, Any
from tabulate import tabulate

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

def parse_fn(example_proto): #type: ignore
    return tf.io.parse_single_example( #type: ignore
        serialized=example_proto,
        features={
            'image/object/class/text': tf.io.VarLenFeature(tf.string) #type: ignore
        }
    ) # type: ignore

def count(infilenames: List[str], displaytotal: bool=False, displaycategories: bool=False) -> None:

    all_categories: Set[str] = set()
    total_cat_counts: Dict[str, int] = {}
    all_file_counts: List[Dict[str, int]] = []
    total: int = 0
    subtotals: List[int] = []
    for infilename in infilenames:
        indataset = tf.data.TFRecordDataset(infilename).map(parse_fn) # type: ignore

        file_cat_counts: Dict[str,int] = {}

        for row in indataset: # type: ignore
            image_cat_counts: Dict[str,int] = {}
            image_categories: List[str] = [c.decode("UTF-8") for c in tf.sparse.to_dense(row['image/object/class/text']).numpy()] #type: ignore

            # Count up all the categories for the image
            for cat in image_categories:
                all_categories.add(cat)
                if image_cat_counts.get(cat) == None:
                    image_cat_counts[cat] = 0
                image_cat_counts[cat] += 1


            # Count up all the categories for the file
            for cat, image_cat_count in image_cat_counts.items():
                if file_cat_counts.get(cat) == None:
                    file_cat_counts[cat] = 0
                file_cat_counts[cat] += image_cat_count

        all_file_counts += [file_cat_counts]

        for cat, file_cat_count in file_cat_counts.items():
            if total_cat_counts.get(cat) == None:
                total_cat_counts[cat] = 0
            total_cat_counts[cat] += file_cat_count

        subtotals += [len([r for r in indataset])] # type: ignore
        total+=subtotals[-1]

    if displaytotal:
        print(total)

    elif displaycategories:
        table: Dict[str,List[Any]] = { 
            'filename': [os.path.basename(f) for f in infilenames],
            'total': subtotals
        }
        table = {**table, **{cat:[] for cat in all_categories}}
        for file_cat_counts in all_file_counts:
            for cat in all_categories:
                if file_cat_counts.get(cat) == None:
                    table[cat] += [0]
                else:
                    table[cat] += [file_cat_counts.get(cat)]

        print(tabulate(table, headers="keys"))

    else:
        for subtotal in subtotals:
            print(subtotal)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display the number of records in each file')
    parser.add_argument('infiles', metavar='infiles', type=str, nargs='+', help='files with records to be counted')
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--total', '-t', action="store_true", help='instead of the the total for each file, display the sum total across all files')
    group.add_argument('--categories', '-c', action="store_true", help='display the number of labels of each category for each file')

    # TODO Create a histogram of the # of labels of a given category per image across all files?
    # OR use the output of the -c flag to input to another tool
    #group.add_argument('--cat-hist', '-C',  type=str, nargs=1, help='create histogram of the number of labels per image for a given category')

    args = parser.parse_args()
    count(args.infiles, args.total, args.categories)