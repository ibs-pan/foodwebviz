import pandas as pd

import pytest

import foodwebs as fw


NODE_DF = pd.DataFrame([
        ('A', True, 1.0, 2.0, 7.0, 1.0),
        ('B', True, 3.0, 1.0, 6.0, 2.0),
        ('C', False, 5.0, 5.0, 5.0, 4.0),
    ], columns=['Names', 'IsAlive', 'Biomass', 'Import', 'Export', 'Respiration'])


FLOW_MATRIX = pd.DataFrame([
    ('A', 0.0, 3.0, 2.0),
    ('B', 0.0, 0.0, 2.0),
    ('C', 0.0, 5, 0.0),
], columns=['Names', 'A', 'B', 'C']).set_index('Names')
FLOW_MATRIX.columns.name = 'Names'


TROPHIC_LEVELS = [1.0, 2.0, 1.0]


BOUNDARIES = ['Import', 'Export', 'Respiration']


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


@pytest.fixture
def food_web():
    return fw.FoodWeb(title='Test', node_df=NODE_DF, flow_matrix=FLOW_MATRIX)


@pytest.fixture
def processed_node_df():
    node_df = NODE_DF.set_index('Names')
    node_df['TrophicLevel'] = TROPHIC_LEVELS
    return node_df.to_dict(orient='index')
