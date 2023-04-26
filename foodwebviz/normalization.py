'''Methods for foodweb's flow normalization.'''
import numpy as np
import networkx as nx


__all__ = [
    'diet_normalization',
    'log_normalization',
    'donor_control_normalization',
    'predator_control_normalization',
    'mixed_control_normalization',
    'tst_normalization'
]


def diet_normalization(foodweb_graph_view):
    '''In this normalization method, each weight is divided by node's diet.
    Diet is sum of all input weights, inlcuding external import.

    Parameters
    ----------
    foodweb_graph_view : networkx.SubGraph
        Graph View representing foodweb

    Returns
    -------
    subgraph : networkx.SubGraph
        Graph View representing normalized foodweb
    '''

    def get_node_diet(node):
        return sum([x[2] for x in foodweb_graph_view.in_edges(node, data='weight')])

    nx.set_edge_attributes(foodweb_graph_view, {(e[0], e[1]): {'weight': e[2] / get_node_diet(e[1])}
                                                for e in foodweb_graph_view.edges(data='weight')})
    return foodweb_graph_view


def log_normalization(foodweb_graph_view):
    '''Normalized weigth is a logarithm of original weight.

    Parameters
    ----------
    foodweb_graph_view : networkx.SubGraph
        Graph View representing foodweb

    Returns
    -------
    subgraph : networkx.SubGraph
        Graph View representing normalized foodweb
    '''
    nx.set_edge_attributes(foodweb_graph_view, {(e[0], e[1]): {'weight': np.log10(e[2])}
                                                for e in foodweb_graph_view.edges(data='weight')})
    return foodweb_graph_view


def donor_control_normalization(foodweb_graph_view):
    '''Each weight is divided by biomass of the "from" node.

    Parameters
    ----------
    foodweb_graph_view : networkx.SubGraph
        Graph View representing foodweb

    Returns
    -------
    subgraph : networkx.SubGraph
        Graph View representing normalized foodweb
    '''
    biomass = nx.get_node_attributes(foodweb_graph_view, "Biomass")
    nx.set_edge_attributes(foodweb_graph_view, {(e[0], e[1]): {'weight': e[2] / biomass[e[0]]}
                                                for e in foodweb_graph_view.edges(data='weight')})
    return foodweb_graph_view


def predator_control_normalization(foodweb_graph_view):
    '''Each weight is divided by biomass of the "to" node.

    Parameters
    ----------
    foodweb_graph_view : networkx.SubGraph
        Graph View representing foodweb

    Returns
    -------
    subgraph : networkx.SubGraph
        Graph View representing normalized foodweb
    '''
    biomass = nx.get_node_attributes(foodweb_graph_view, "Biomass")
    nx.set_edge_attributes(foodweb_graph_view, {(e[0], e[1]): {'weight': e[2] / biomass[e[1]]}
                                                for e in foodweb_graph_view.edges(data='weight')})
    return foodweb_graph_view


def mixed_control_normalization(foodweb_graph_view):
    '''Each weight is equal to donor_control * predator_control.

    Parameters
    ----------
    foodweb_graph_view : networkx.SubGraph
        Graph View representing foodweb

    Returns
    -------
    subgraph : networkx.SubGraph
        Graph View representing normalized foodweb
    '''
    biomass = nx.get_node_attributes(foodweb_graph_view, "Biomass")
    nx.set_edge_attributes(foodweb_graph_view, {(e[0], e[1]):
                                                {'weight': (e[2] / biomass[e[0]]) * (e[2] / biomass[e[1]])}
                                                for e in foodweb_graph_view.edges(data='weight')})
    return foodweb_graph_view


def tst_normalization(foodweb_graph_view):
    '''Function returning a list of internal flows normalized to TST.

    Parameters
    ----------
    foodweb_graph_view : networkx.SubGraph
        Graph View representing foodweb

    Returns
    -------
    subgraph : networkx.SubGraph
        Graph View representing normalized foodweb
    '''
    tst = sum([x[2] for x in foodweb_graph_view.edges(data='weight')])
    nx.set_edge_attributes(foodweb_graph_view, {(e[0], e[1]): {'weight': e[2] / tst}
                                                for e in foodweb_graph_view.edges(data='weight')})
    return foodweb_graph_view


def normalization_factory(foodweb_graph_view, norm_type):
    '''Applies apropiate normalization method according to norm_type argument.

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
        'donor_control': donor_control_normalization,
        'predator_control': predator_control_normalization,
        'mixed_control': mixed_control_normalization,
        'log': log_normalization,
        'diet': diet_normalization,
        'tst': tst_normalization
    }

    if norm_type == 'linear':
        return foodweb_graph_view

    if norm_type and norm_type.lower() in normalization_methods:
        return normalization_methods[norm_type.lower()](foodweb_graph_view)
    return foodweb_graph_view
