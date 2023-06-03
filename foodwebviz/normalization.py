'''Methods for foodweb's flow normalization.'''
import numpy as np
import networkx as nx


__all__ = [
    'normalization_factory'
]


class LinearNormalization(object):
    '''
    Dummy normalization.
    '''

    def __init__(self, foodweb_graph_view: nx.Graph) -> None:
        self.foodweb_graph_view = foodweb_graph_view

    def normalize(self) -> nx.Graph:
        edges = {}
        for u, v, data in self.foodweb_graph_view.edges(data=True):
            edges[(u, v)] = {'weight': self._norm_func(u, v, data)}

        nx.set_edge_attributes(self.foodweb_graph_view, edges)
        return self.foodweb_graph_view

    def _norm_func(self, node_from: str, node_to: str , data: dict[str, float]) -> float:
        return data['weight']


class DietNormalization(LinearNormalization):
    '''
    In this normalization method, each weight is divided by node's diet.
    Diet is sum of all input weights, inlcuding external import.
    '''

    def _get_node_diet(self, node):
        return sum(data['weight'] for _, _, data in self.foodweb_graph_view.in_edges(node, data=True))

    def _norm_func(self, node_from, node_to, data):
        return data['weight'] / self._get_node_diet(node_to)


class LogNormalization(LinearNormalization):
    '''
    Normalized weigth is a logarithm of original weight.
    '''

    def _norm_func(self, node_from, node_to, data):
        return np.log10(data['weight'])


class DonorControlNormalization(LinearNormalization):
    '''
    Each weight is divided by biomass of the "from" node.
    '''

    def __init__(self, foodweb_graph_view) -> None:
        super().__init__(foodweb_graph_view)
        self.biomass = nx.get_node_attributes(foodweb_graph_view, "Biomass")

    def _norm_func(self, node_from, node_to, data):
        return data['weight'] / self.biomass[node_from]


class PredatorControlNormalization(LinearNormalization):
    '''
    Each weight is divided by biomass of the "to" node.
    '''

    def __init__(self, foodweb_graph_view) -> None:
        super().__init__(foodweb_graph_view)
        self.biomass = nx.get_node_attributes(foodweb_graph_view, "Biomass")

    def _norm_func(self, node_from, node_to, data):
        return data['weight'] / self.biomass[node_to]


class MixedControlNormalization(LinearNormalization):
    '''
    Each weight is equal to donor_control * predator_control.
    '''

    def __init__(self, foodweb_graph_view) -> None:
        super().__init__(foodweb_graph_view)
        self.biomass = nx.get_node_attributes(foodweb_graph_view, "Biomass")

    def _norm_func(self, node_from, node_to, data):
        return (data['weight'] / self.biomass[node_from]) * (data['weight'] / self.biomass[node_to])


class TSTNormalization(LinearNormalization):
    '''
    Internal flows are normalized to TST.
    '''

    def __init__(self, foodweb_graph_view) -> None:
        super().__init__(foodweb_graph_view)
        self.tst = sum([data['weight'] for _, _, data in foodweb_graph_view.edges(data=True)])

    def _norm_func(self, node_from, node_to, data):
        return data['weight'] / self.tst


def normalization_factory(foodweb_graph_view, norm_type):
    '''
    Applies apropiate normalization method according to norm_type argument.

    Parameters
    ----------
    foodweb_graph_view : networkx.SubGraph
        Graph View representing foodweb
    norm_type : string
        Represents normalization type to use.
        Available options are: 'diet', 'log', 'biomass', and 'tst'.

    Returns
    -------
    subgraph : networkx.SubGraph
        Graph View representing normalized foodweb
    '''
    normalization_methods = {
        'donor_control': DonorControlNormalization,
        'predator_control': PredatorControlNormalization,
        'mixed_control': MixedControlNormalization,
        'log': LogNormalization,
        'diet': DietNormalization,
        'tst': TSTNormalization,
        'linear': LinearNormalization
    }

    if not norm_type:
        return foodweb_graph_view

    normalization_cls = normalization_methods.get(norm_type.lower())

    if normalization_cls:
        return normalization_cls(foodweb_graph_view).normalize()
    return foodweb_graph_view
