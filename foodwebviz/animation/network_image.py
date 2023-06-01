# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:43:31 2021
Class defining the image of network - positions of nodes, sizes of nodes,
density of flow and other parameters independent of the way the animation is implemented.
@author: Mateusz
"""

import sys
import numpy as np
import pandas as pd
import itertools
import random

from numpy.random import uniform

from foodwebviz.utils import squeeze_map


class NetworkImage(object):
    '''
    Class defining the image of network:
    positions of nodes, sizes of nodes, density of flow
    and other parameters independent of the way the animation is implemented.
    '''

    def __init__(self, net, with_detritus=False, k_=20, min_part_num=2, map_fun=np.log10, max_part=1000):
        '''Initialize a netImage from a foodweb net.
            Parameters
            ----------
            title : string
                Name of the foodweb.
            nodes : pandas.DataFrame with node attributes-
                - positions of nodes in canvas [0,100] x [0,100]
                - node names
                - node biomasses
            particleNumbers : [pandas.DataFrame with numbers of particles flowing
                from row node to column node, particles moving in imports, outflows to environment]
        '''
        self.title = net.title
        self.nodes = self._get_node_attributes(net)
        self.particle_numbers = self._get_particle_numbers(
            net=net,
            with_detritus=with_detritus,
            min_part_num=min_part_num,
            map_fun=map_fun,
            max_part=max_part)

        # optimize node positions using Fruchterman - Rheingold algorithm
        self.nodes[['x', 'y']] = self._fruchterman_reingold_layout(
            A=net.flow_matrix.applymap(lambda x: float(x > 0.0)),
            dim=2,
            k=k_,
            pos=self.nodes[['x', 'y']].values,
            iterations=20,
            hold_dim=1,
            hard_spheres=False,
            if_only_attraction=True)

        self.nodes[['x', 'y']] = self._fruchterman_reingold_layout(
            A=net.flow_matrix.applymap(lambda x: float(x > 0.0)),
            dim=2,
            k=k_,
            pos=self.nodes[['x', 'y']].values,
            iterations=50,
            hold_dim=1,
            hard_spheres=True,
            if_only_attraction=False)

    def _aggregate_in_trophic_level_layers(self, pos_df):
        def assign_rank(df):
            current, ranks = 0, []
            for i, bin in enumerate(df.TrophicLevel_bin.values):
                if i > 1 and bin == df.TrophicLevel_bin.values[i - 1]:
                    current += 1
                else:
                    current = 1
                ranks.append(current)
            return ranks

        grouped = pos_df.groupby('TrophicLevel_bin').count()
        grouped['odd_or_even'] = [x % 2 == 1 for x in range(1, len(grouped) + 1)]
        pos_df['x_rank'] = assign_rank(pos_df)
        # get the width of trophic layer for each node
        pos_df['odd_or_even_layer'] = pos_df.apply(lambda x: grouped.loc[x.TrophicLevel_bin, 'odd_or_even'],
                                                   axis='columns')

        # return regularly spaced x positions for num number of items
        max_width = grouped['width'].max()
        grouped['xs'] = grouped.apply(lambda x: list(np.linspace(10 + 5 * (max_width - x.TrophicLevel),
                                                                 90 - 5 * (max_width-x.TrophicLevel),
                                                                 num=x.TrophicLevel)),
                                      axis='columns')
        pos_df = pos_df.sort_values(by='out_to_living')
        return grouped

    def _get_node_attributes(self, net):
        # function sets initial node positions before an interative layout algorithm will optimise them
        # node positions are based upon node properties, here on trophic levels
        pos_df = net.node_df[['TrophicLevel', 'Biomass']].reset_index().set_index('Names', drop=False).copy()

        # assuming 5 nodes per band on average, we get the number of bins
        trophic_levels_bins = int(net.n / 5)+1

        # we divide trophic levels into bins
        pos_df['TrophicLevel_bin'] = (
            pd.cut(x=pos_df['TrophicLevel'], bins=trophic_levels_bins)
            .apply(lambda x: x.mid)
        )

        pos_df['out_to_living'] = net.get_outflows_to_living()

        pos_df['y'] = np.interp(pos_df['TrophicLevel'],
                                (pos_df['TrophicLevel'].min(), pos_df['TrophicLevel'].max()),
                                (5, 95))

        # number of nodes that have TL +- 0.5
        pos_df['width'] = pos_df['TrophicLevel'].apply(
            lambda x: len(pos_df[np.abs(pos_df['TrophicLevel'] - x) < 0.25]))

        # select horizontal node positions with minimal number of intersections between flows
        pos_df = self._find_minimal_intersections(net, pos_df)

        # move the nodes at random a bit
        pos_df['x'] = pos_df['x'] + uniform(-8, 8, len(pos_df))
        return pos_df

    def _find_minimal_intersections(self, net, pos_df):
        '''
        given nodes grouped within layers of trophic levels it
        first generates a random but regular node placement in x variable, then
        computes the number of intersections for each and chooses the one with the smallest
        '''
        def randomly_choose_x(row, grouped):
            # assign the next free x from the queue of available values in pos_df
            random_x = random.choice(range(len(grouped.loc[row['TrophicLevel_bin'], 'xs'])))
            return grouped.loc[row['TrophicLevel_bin'], 'xs'].pop(random_x)

        min_intersections = sys.maxsize
        best_pos = None
        for _ in range(20):
            grouped = self._aggregate_in_trophic_level_layers(pos_df)
            pos_df['x'] = pos_df.apply(lambda x: randomly_choose_x(x, grouped=grouped), axis='columns')
            intersections = self._get_num_of_crossed_edges(net, pos_df)

            if intersections <= min_intersections:
                min_intersections = intersections
                best_pos = pos_df.copy()
        return best_pos

    def _get_particle_numbers(self, net, with_detritus, min_part_num=2, map_fun=np.log10, max_part=1000):
        '''
        min_part_num: fixed number of particles for a minimal flow
        '''

        # tu jednak trzeba zerować, a nie usuwać
        flows = net.get_flow_matrix(boundary=False, to_alive_only=~with_detritus)

        # the minimal flow will correspond to a fixed number of particles
        min_flow = flows.values[flows.values > 0].min()
        # the maximal flow will correspond to the maximum number of particles we can handle
        max_flow = flows.values[flows.values > 0].max()

        def calc_particle_number(x): return int(
            squeeze_map(x, min_flow, max_flow, map_fun, min_part_num, max_part)) if x != 0.0 else 0

        return [flows.applymap(calc_particle_number),
                net.node_df.Import.apply(calc_particle_number),
                (net.node_df.Export + net.node_df.Respiration).apply(calc_particle_number)]

    def _fruchterman_reingold_layout(self, A, dim=2, k=None, pos=None, fixed=None,
                                     iterations=100, hold_dim=None, min_dist=0.01,
                                     hard_spheres=True, if_only_attraction=False):
        '''
        Position nodes in adjacency matrix A using Fruchterman-Reingold
        Entry point for NetworkX graph is fruchterman_reingold_layout()

        Following by Marscher, adapted from NetworkX pyemma/plots/_ext/fruchterman_reingold.py
        https://github.com/markovmodel/PyEMMA/blob/58825588431020d7e2a2ea57a941abc86647fc0e/pyemma/plots/_ext/fruchterman_reingold.py
        '''

        try:
            nnodes, _ = A.shape
        except AttributeError:
            raise RuntimeError(
                "fruchterman_reingold() takes an adjacency matrix as input")

        A = np.asarray(A)  # make sure we have an array instead of a matrix

        if pos is None:
            # random initial positions
            pos = np.asarray(np.random.random((nnodes, dim)), dtype=A.dtype)
        else:
            # make sure positions are of same type as matrix
            pos = pos.astype(A.dtype)

        # optimal distance between nodes
        if k is None:
            k = np.sqrt(1.0 / nnodes)
        # the initial "temperature"  is about .1 of domain area (=1x1)
        # this is the largest step allowed in the dynamics.
        t = 0.1
        # simple cooling scheme.
        # linearly step down by dt on each iteration so last iteration is size dt.
        dt = t / float(iterations + 1)
        delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
        # the inscrutable (but fast) version
        # this is still O(V^2)
        # could use multilevel methods to speed this up significantly
        for _ in range(iterations):
            # matrix of difference between points
            for i in range(pos.shape[1]):
                delta[:, :, i] = pos[:, i, None] - pos[:, i]
            dist_from_center_x = pos[:, 0] - np.full(nnodes, 50)
            # distance between points
            distance = np.sqrt((delta**2).sum(axis=-1))
            # enforce minimum distance of min_dist
            distance = np.where(distance < min_dist, min_dist, distance)

            # adding a hard spheres potential - to keep nodes further apart
            if hard_spheres:
                sphere = distance * 100 * np.exp(10 * (10 - distance))
            else:
                sphere = np.zeros((distance.shape[0], distance.shape[1]), dtype=float)

            # displacement "force"
            if if_only_attraction:
                force = - A * distance / k
            else:
                centrifugal = 100 * dist_from_center_x**2
                force = k * k / distance**2 - A * distance / k + sphere + centrifugal

            displacement = np.transpose(np.transpose(delta) * force).sum(axis=1)

            # update positions
            length = np.sqrt((displacement**2).sum(axis=1))
            length = np.where(length < min_dist, 0.1, length)
            delta_pos = np.transpose(np.transpose(displacement) * t / length)

            if fixed is not None:
                # don't change positions of fixed nodes
                delta_pos[fixed] = 0.0

            # only update y component
            if hold_dim == 0:
                pos[:, 1] += delta_pos[:, 1]
            # only update x component
            elif hold_dim == 1:
                pos[:, 0] += delta_pos[:, 0]
            else:
                pos += delta_pos

            # cool temperature
            t -= dt

            pos = self._rescale_layout(pos, [[8, 92], [10, 95]])
        return pos

    def _rescale_layout(self, pos, borders):
        # rescale to (origin_,scale) in all axes

        # shift origin to (origin_,origin_)
        lim = np.zeros(pos.shape[1])  # max coordinate for all axes
        for i in range(pos.shape[1]):
            pos[:, i] -= pos[:, i].min()
            lim[i] = max(pos[:, i].max(), lim[i])
        # rescale to (0,scale) in all directions, preserves aspect
        for i in range(pos.shape[1]):
            pos[:, i] = np.interp(pos[:, i], [0, lim[i]], borders[i])
        return pos

    def _get_num_of_crossed_edges(self, net, positions):
        '''
        returns the approximate number of crossing between edges given the positions of their ends
        approximate in order not to solve line equation -> but this does not have to prolong
        evaluation time greatly
        '''
        edges = net.get_flows(boundary=False,
                              mark_alive_nodes=False,
                              normalization=None,
                              no_flows_to_detritus=True)

        # get coordinates of a link as a list of ((x1,y1),(x2,y2))
        links = [tuple(positions.loc[[node_1, node_2], ['x', 'y']].itertuples(index=False, name=None))
                 for node_1, node_2, _ in edges if node_1 != node_2]
        return sum([is_intersect(pair) for pair in itertools.combinations(links, 2)])


def is_intersect(pair):
    '''
    Helper function that returns true if the line segment 'p1q1' and 'p2q2' intersect.
    From https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    '''
    def on_segment(p, q, r):
        return (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and \
            (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))

    def orientation(p, q, r):
        '''
        to find the orientation of an ordered triplet (p,q,r)
        function returns the following values:
        0 : Colinear points
        1 : Clockwise points
        2 : Counterclockwise

        See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/
        for details of below formula.
        '''
        val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
        if val > 0:
            return 1  # Clockwise orientation
        elif val < 0:
            return 2  # Counterclockwise orientation
        return 0  # Colinear orientation

    (p1, p2), (q1, q2) = pair

    # Find the 4 orientations required for
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases

    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0) and on_segment(p1, p2, q1):
        return True

    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
    if (o2 == 0) and on_segment(p1, q2, q1):
        return True

    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0) and on_segment(p2, p1, q2):
        return True

    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0) and on_segment(p2, q1, q2):
        return True

    # If none of the cases
    return False


if __name__ == "__main__":
    import foodwebviz as fw

    scor_file_in = 'Alaska_Prince_William_Sound.scor'  # 'atlss_cypress_wet.dat'

    foodweb = fw.read_from_SCOR(scor_file_in)
    netIm = NetworkImage(foodweb, False, k_=80, min_part_num=1, map_fun=np.sqrt, max_part=20)
    print(netIm.nodes)
