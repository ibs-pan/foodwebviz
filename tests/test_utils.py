import numpy as np

import foodwebviz as fw

from trophic_levels import TROPHIC_LEVELS_REAL # TODO

def test_calc_trophic_levels(food_web):
    np.testing.assert_almost_equal(np.array([1.0, 2.0, 1.0]), fw.calculate_trophic_levels(food_web), 4)


# def test_calc_trophic_levels2(self):
#     import glob
#     from foodwebs.foodweb_io import readFW_SCOR

#     food_webs = [readFW_SCOR(net_path) for net_path in glob.glob('./data/*')]
#     for web, trophic in zip(food_webs, TROPHIC_LEVELS_REAL):
#         web_trophic = web._calcTrophicLevels()
#         np.testing.assert_almost_equal(trophic, web_trophic, 7)
