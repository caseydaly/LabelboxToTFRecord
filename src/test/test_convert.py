# vim:ts=4:sw=4:sts=4

import sys
import unittest
import convert

# TODO convert to pytest
class TestConvert(unittest.TestCase):
    def test_split_small_list(self):
        result = convert.splits_to_record_indices([20, 30, 50], 2)
        self.assertEqual(result, [1, 2])

    def test_split_single(self):
        result = convert.splits_to_record_indices([100], 1000)
        self.assertEqual(result, [1000])

    def test_split_empty(self):
        result = convert.splits_to_record_indices([], 1000)
        self.assertEqual(result, [1000])

        result = convert.splits_to_record_indices(None, 1000)
        self.assertEqual(result, [1000])

    def test_split_happy(self):
        result = convert.splits_to_record_indices([20, 30, 50], 100)
        self.assertEqual(result, [20, 50, 100])

    def test_split_ten(self):
        result = convert.splits_to_record_indices([20, 30, 50], 10)
        self.assertEqual(result, [2, 5, 10])

    def test_split_last_split_rounded_down(self):
        result = convert.splits_to_record_indices([30, 70], 675)
        self.assertEqual(result, [202, 675])
