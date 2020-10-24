import unittest

import pandas as pd
import numpy as np

from foodwebs.foodweb import FoodWeb
from foodwebs.foodweb_io import read_from_SCOR
from trophic_levels import TROPHIC_LEVELS_REAL


NODE_DF = pd.DataFrame([
    ('A', True, 1.0, 2.0, 7.0, 1.0),
    ('B', True, 3.0, 1.0, 6.0, 2.0),
    ('C', False, 5.0, 5.0, 5.0, 4.0),
], columns=['Names', 'IsAlive', 'Biomass', 'Import', 'Export', 'Respiration'])

TROPHIC_LEVELS = [1.0, 2.0, 1.0]

FLOW_MATRIX = pd.DataFrame([
    ('A', 0.0, 3.0, 2.0),
    ('B', 0.0, 0.0, 2.0),
    ('C', 0.0, 5, 0.0),
], columns=['Names', 'A', 'B', 'C']).set_index('Names')
FLOW_MATRIX.columns.name = 'Names'

FLOWS = [
    ('A', 'B', {'weight': 3.0}),
    ('A', 'C', {'weight': 2.0}),
    ('A', 'Export', {'weight': 7.0}),
    ('A', 'Respiration', {'weight': 1.0}),
    ('B', 'C', {'weight': 2.0}),
    ('B', 'Export', {'weight': 6.0}),
    ('B', 'Respiration', {'weight': 2.0}),
    ('C', 'B', {'weight': 5.0}),
    ('C', 'Export', {'weight': 5.0}),
    ('C', 'Respiration', {'weight': 4.0}),
    ('Import', 'A', {'weight': 2.0}),
    ('Import', 'B', {'weight': 1.0}),
    ('Import', 'C', {'weight': 5.0})
]

BOUNDARIES = ['Import', 'Export', 'Respiration']


def _get_processed_node_df():
    node_df = NODE_DF.set_index('Names')
    node_df['TrophicLevel'] = TROPHIC_LEVELS
    return node_df


class FoodWebTestCase(unittest.TestCase):
    def test_init_foodweb(self):
        web = FoodWeb(title='Test', node_df=NODE_DF, flow_matrix=FLOW_MATRIX)

        self.assertEqual(web.n, 3)
        self.assertEqual(web.n_living, 2)
        self.assertEqual(web.get_links_number(), 4)

    def test_getFlowMatWithBoundary(self):
        web = FoodWeb(title='Test', node_df=NODE_DF, flow_matrix=FLOW_MATRIX)
        flow_matrix_with_boundary = web.get_flow_matrix(boundary=True)

        self.assertTrue(all(col in flow_matrix_with_boundary.columns for col in BOUNDARIES))
        self.assertTrue(all(col in flow_matrix_with_boundary.index for col in BOUNDARIES))

        for b in BOUNDARIES:
            self.assertEqual(flow_matrix_with_boundary[b].loc[b], 0.0)

    def test_graph(self):
        web = FoodWeb(title='Test', node_df=NODE_DF, flow_matrix=FLOW_MATRIX)

        web_nodes = dict(web._graph.nodes(data=True))
        test_nodes = _get_processed_node_df().to_dict(orient='index')
        for b in BOUNDARIES:
            test_nodes[b] = {}

        self.assertDictEqual(web_nodes, test_nodes)
        self.assertListEqual(list(web._graph.edges(data=True)), FLOWS)

    def test_get_graph(self):
        web = FoodWeb(title='Test', node_df=NODE_DF, flow_matrix=FLOW_MATRIX)

        # no boundaries
        web_nodes = web.get_graph(boundary=False, mark_alive_nodes=False).nodes()
        test_nodes = _get_processed_node_df().index
        self.assertListEqual(list(web_nodes), list(test_nodes))

        # with boundaries
        web_nodes = web.get_graph(boundary=True, mark_alive_nodes=False).nodes()
        self.assertListEqual(list(web_nodes), list(test_nodes) + BOUNDARIES)

        # with not alive marks
        web_nodes = web.get_graph(boundary=False, mark_alive_nodes=True).nodes()
        self.assertListEqual(list(web_nodes), ['A', 'B', '✗ C'])

    def test_calc_trophic_levels(self):
        web = FoodWeb(title='Test', node_df=NODE_DF, flow_matrix=FLOW_MATRIX)
        np.testing.assert_almost_equal(np.array([1.0, 2.0, 1.0]), web._calculate_trophic_levels(), 4)

    # def test_calc_trophic_levels2(self):
    #     import glob
    #     from foodwebs.foodweb_io import readFW_SCOR

    #     food_webs = [readFW_SCOR(net_path) for net_path in glob.glob('./data/*')]
    #     for web, trophic in zip(food_webs, TROPHIC_LEVELS_REAL):
    #         web_trophic = web._calcTrophicLevels()
    #         np.testing.assert_almost_equal(trophic, web_trophic, 7)

    def test_get_norm_intern_flows(self):
        normalized_weights = {
            'tst': [0.25, 0.16666666666666666, 0.16666666666666666, 0.4166666666666667],
            'biomass': [3.0, 2.0, 0.6666666666666666, 1.0],
            'log': [1.0986122886681098, 0.6931471805599453, 0.6931471805599453, 1.6094379124341003],
            'diet': [0.375, 0.5, 0.5, 0.625]
        }

        web = FoodWeb(title='Test', node_df=NODE_DF, flow_matrix=FLOW_MATRIX)
        for norm, expected in normalized_weights.items():
            weights = [e[2]['weight'] for e in web.get_flows(normalization=norm)]
            self.assertListEqual(weights, expected)

        g = web.get_graph(boundary=True, normalization='diet')
        for node in g.nodes():
            # sum of each node's input weights should be 1
            if node != 'Import':
                self.assertEqual(sum([x[2] for x in g.in_edges(node, data='weight')]), 1.0)

    def test_write_scor(self):
        web = FoodWeb(title='Test', node_df=NODE_DF, flow_matrix=FLOW_MATRIX)

        scor_filename = 'test.scor'
        web.write_SCOR(scor_filename)

        web_from_file = read_from_SCOR(scor_filename)
        self.assertEqual(web.n, web_from_file.n)
        self.assertEqual(web.n_living, web_from_file.n_living)
        pd.testing.assert_frame_equal(web.flow_matrix, web_from_file.flow_matrix)
        pd.testing.assert_frame_equal(web.node_df, web_from_file.node_df)


if __name__ == '__main__':
    unittest.main()
