
import numpy as np
import networkx as nx


def diet_normalization(graph_view):
    '''
    In this normalization method, each weight is divided by node's diet,
    which is sum of it's input weights, inlcuding Import
    '''
    def get_node_diet(node):
        return sum([x[2] for x in graph_view.in_edges(node, data='weight')])

    nx.set_edge_attributes(graph_view, {(e[0], e[1]): {'weight': e[2] / get_node_diet(e[1])}
                                        for e in graph_view.edges(data='weight')})
    return graph_view


def log_normalization(graph_view):
    nx.set_edge_attributes(graph_view, {(e[0], e[1]): {'weight': np.log10(e[2])}
                                        for e in graph_view.edges(data='weight')})
    return graph_view


def biomass_normalization(graph_view):
    biomass = nx.get_node_attributes(graph_view, "Biomass")
    nx.set_edge_attributes(graph_view, {(e[0], e[1]): {'weight': e[2] / biomass[e[0]]}
                                        for e in graph_view.edges(data='weight')})
    return graph_view


def tst_normalization(graph_view):
    '''Function returning a list of internal flows normalized to TST'''

    TST = sum([x[2] for x in graph_view.edges(data='weight')])
    nx.set_edge_attributes(graph_view, {(e[0], e[1]): {'weight': e[2] / TST}
                                        for e in graph_view.edges(data='weight')})
    return graph_view


NORMALIZATION = {
    'biomass': biomass_normalization,
    'log': log_normalization,
    'diet': diet_normalization,
    'tst': tst_normalization
}


def flows_normalization(graph_view, norm_type):
    if norm_type in NORMALIZATION:
        return NORMALIZATION[norm_type](graph_view)
    return graph_view
