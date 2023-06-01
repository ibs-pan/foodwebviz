import numpy as np

import foodwebviz as fw


def test_calc_trophic_levels(food_web):
    np.testing.assert_almost_equal(np.array([1.0, 2.0, 1.0]), fw.calculate_trophic_levels(food_web), 4)
