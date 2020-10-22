import unittest

import pandas as pd

from foodwebs.normalization import diet_normalization


FLOWS_PD = pd.DataFrame([
    ('Import', 'A', 2, 0),
    ('Import', 'B', 1, 0),
    ('Import', 'C', 5, 0),
    ('A', 'B', 3, 0),
    ('A', 'C', 2, 0),
    ('A', 'Export', 7, 3),
    ('A', 'Respiration', 1, 0),
    ('A', 'Export', 7, 0),
    ('B', 'C', 2, 0),
    ('B', 'Export', 6, 7),
    ('B', 'Respiration', 2, 0),
    ('C', 'B', 5, 0),
    ('C', 'Export', 5, 0),
    ('C', 'Respiration', 4, 0),
], columns=['from', 'to', 'weights', 'extra_col'])


# class NormalizationTestCase(unittest.TestCase):
#     def test_diet_normalization(self):
#         res_pd = diet_normalization(FLOWS_PD)

#         # test if every column sums up to 1
#         self.assertSetEqual(set(res_pd.groupby('to').sum()['weights'].values), set([1]))


if __name__ == '__main__':
    unittest.main()
