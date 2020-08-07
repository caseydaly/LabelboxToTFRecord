# vim:ts=4:sw=4:sts=4

import sys
import unittest
import convert

class TestConvert(unittest.TestCase):
#[20,30], 10
#[20,30], 100
#[50,30,20], 2
#[1,99], 2
    def test_split_small_list(self):
        result = convert.splits_to_record_indices([20, 30, 50], 2)
        self.assertEqual(result, [1])

    def test_split_single(self):
        result = convert.splits_to_record_indices([100], 1000)
        self.assertEqual(result, [1000])

    def test_split_empty(self):
        result = convert.splits_to_record_indices([], 1000)
        self.assertEqual(result, [1000])

        result = convert.splits_to_record_indices(None, 1000)
        self.assertEqual(result, [1000])

    def test_split_happy(self):
        result = convert.splits_to_record_indices([20, 30], 100)
        self.assertEqual(result, [20, 50, 100])
