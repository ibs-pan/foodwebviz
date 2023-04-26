'''Methods for foodweb's flow normalisation.'''
import numpy as np
import networkx as nx


__all__ = [
    'diet_normalisation',
    'log_normalisation',
    'donor_control_normalisation',
    'predator_control_normalisation',
    'mixed_control_normalisation',
    'tst_normalisation'
]


def diet_normalisation(foodweb_graph_view):
    '''In this normalisation method, each weight is divided by node's diet.
    Diet is sum of all input weights, inlcuding external import.

    Parameters
    ----------
    foodweb_graph_view : networkx.SubGraph
        Graph View representing foodweb

    Returns
    -------
    subgraph : networkx.SubGraph
        Graph View representing normalised foodweb
    '''

    def get_node_diet(node):
        return sum([x[2] for x in foodweb_graph_view.in_edges(node, data='weight')])

    nx.set_edge_attributes(foodweb_graph_view, {(e[0], e[1]): {'weight': e[2] / get_node_diet(e[1])}
                                                for e in foodweb_graph_view.edges(data='weight')})
    return foodweb_graph_view


def log_normalisation(foodweb_graph_view):
    '''normalised weigth is a logarithm of original weight.

    Parameters
    ----------
    foodweb_graph_view : networkx.SubGraph
        Graph View representing foodweb

    Returns
    -------
    subgraph : networkx.SubGraph
        Graph View representing normalised foodweb
    '''
    nx.set_edge_attributes(foodweb_graph_view, {(e[0], e[1]): {'weight': np.log10(e[2])}
                                                for e in foodweb_graph_view.edges(data='weight')})
    return foodweb_graph_view


def donor_control_normalisation(foodweb_graph_view):
    '''Each weight is divided by biomass of the "from" node.

    Parameters
    ----------
    foodweb_graph_view : networkx.SubGraph
        Graph View representing foodweb

    Returns
    -------
    subgraph : networkx.SubGraph
        Graph View representing normalised foodweb
    '''
    biomass = nx.get_node_attributes(foodweb_graph_view, "Biomass")
    nx.set_edge_attributes(foodweb_graph_view, {(e[0], e[1]): {'weight': e[2] / biomass[e[0]]}
                                                for e in foodweb_graph_view.edges(data='weight')})
    return foodweb_graph_view


def predator_control_normalisation(foodweb_graph_view):
    '''Each weight is divided by biomass of the "to" node.

    Parameters
    ----------
    foodweb_graph_view : networkx.SubGraph
        Graph View representing foodweb

    Returns
    -------
    subgraph : networkx.SubGraph
        Graph View representing normalised foodweb
    '''
    biomass = nx.get_node_attributes(foodweb_graph_view, "Biomass")
    nx.set_edge_attributes(foodweb_graph_view, {(e[0], e[1]): {'weight': e[2] / biomass[e[1]]}
                                                for e in foodweb_graph_view.edges(data='weight')})
    return foodweb_graph_view


def mixed_control_normalisation(foodweb_graph_view):
    '''Each weight is equal to donor_control * predator_control.

    Parameters
    ----------
    foodweb_graph_view : networkx.SubGraph
        Graph View representing foodweb

    Returns
    -------
    subgraph : networkx.SubGraph
        Graph View representing normalised foodweb
    '''
    biomass = nx.get_node_attributes(foodweb_graph_view, "Biomass")
    nx.set_edge_attributes(foodweb_graph_view, {(e[0], e[1]):
                                                {'weight': (e[2] / biomass[e[0]]) * (e[2] / biomass[e[1]])}
                                                for e in foodweb_graph_view.edges(data='weight')})
    return foodweb_graph_view


def tst_normalisation(foodweb_graph_view):
    '''Function returning a list of internal flows normalised to TST.

    Parameters
    ----------
    foodweb_graph_view : networkx.SubGraph
        Graph View representing foodweb

    Returns
    -------
    subgraph : networkx.SubGraph
        Graph View representing normalised foodweb
    '''
    tst = sum([x[2] for x in foodweb_graph_view.edges(data='weight')])
    nx.set_edge_attributes(foodweb_graph_view, {(e[0], e[1]): {'weight': e[2] / tst}
                                                for e in foodweb_graph_view.edges(data='weight')})
    return foodweb_graph_view


def normalisation_factory(foodweb_graph_view, norm_type):
    '''Applies apropiate normalisation method according to norm_type argument.

    Parameters
    ----------
    foodweb_graph_view : networkx.SubGraph
        Graph View representing foodweb
    norm_type : string
        Represents normalisation type to use.
        Available options are: 'diet', 'log', 'biomass', and 'tst'.

    Returns
    -------
    subgraph : networkx.SubGraph
        Graph View representing normalised foodweb
    '''
    normalisation_methods = {
        'donor_control': donor_control_normalisation,
        'predator_control': predator_control_normalisation,
        'mixed_control': mixed_control_normalisation,
        'log': log_normalisation,
        'diet': diet_normalisation,
        'tst': tst_normalisation
    }

    if norm_type == 'linear':
        return foodweb_graph_view

    if norm_type.lower() in normalisation_methods:
        return normalisation_methods[norm_type.lower()](foodweb_graph_view)
    
    return foodweb_graph_view