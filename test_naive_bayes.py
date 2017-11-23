from unittest import TestCase
import unittest
from gauss_nb import GaussNB


class GaussNBTest(TestCase):

    def setUp(self):
        self.nb = GaussNB()

    def test_split_data(self):
        dataset = [[1], [2], [3], [4], [5]]
        splitRatio = 0.70
        train, test = self.nb.split_data(dataset, splitRatio)
        self.assertEqual(3, len(train))
        self.assertEqual(2, len(test))

if __name__ == '__main__':
    unittest.main()