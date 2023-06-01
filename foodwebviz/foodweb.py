'''Class for foodwebs.'''
import networkx as nx

import foodwebviz as fw
from .normalization import normalization_factory


__all__ = [
    'FoodWeb'
]


class FoodWeb(object):
    '''
    Class defining a food web of an ecosystem.
    It stores species and flows between them with additional data like Biomass.
    '''

    def __init__(self, title, node_df, flow_matrix):
        '''Initialize a foodweb with title, nodes and flow matrix.
            Parameters
            ----------
            title : string
                Name of the foodweb.
            node_df : pd.DataFrame
                Species data respresented in a set of the following columns:
                ['Names', 'IsAlive', 'Biomass', 'Import', 'Export', 'Respiration']
            flow_matrix : pd.DataFrame
                Data containing list of flows between species, adjacency matrix,
                where the intersection between ith column and jth row represents
                flow from node i to j.
            See Also
            --------
            io.read_from_SCOR
        '''
        self.title = title
        self.node_df = node_df.set_index("Names")
        self.flow_matrix = flow_matrix

        self.n = len(self.node_df)
        self.n_living = len(self.node_df[self.node_df.IsAlive])

        if len(flow_matrix) > 1:
            self.node_df['TrophicLevel'] = fw.calculate_trophic_levels(self)
        self._graph = self._init_graph()

    def _init_graph(self):
        '''Returns networkx.DiGraph initialized using foodweb's flow matrix.'''
        graph = nx.from_pandas_adjacency(self.get_flow_matrix(boundary=True),  create_using=nx.DiGraph)
        nx.set_node_attributes(graph, self.node_df.to_dict(orient='index'))

        exclude_edges = []
        for n in self.node_df.index.values:
            exclude_edges.append((n, 'Import'))
            exclude_edges.append(('Export', n))
            exclude_edges.append(('Respiration', n))
        graph.remove_edges_from(exclude_edges)
        return graph

    def get_diet_matrix(self):
        '''Returns a matrix of system flows express as diet proportions=
        =fraction of node inflows this flow contributes'''
        return self.flow_matrix.div(self.flow_matrix.sum(axis=0), axis=1).fillna(0.0)

    def get_graph(self, boundary=False, mark_alive_nodes=False, normalization=None,
                  no_flows_to_detritus=False):
        '''Returns foodweb as networkx.SubGraph View fo networkx.DiGraph.

        Parameters
        ----------
        boundary : bool, optional (default=False)
            If True, boundary flows will be added to the graph.
            Boundary flows are: Import, Export, and Repiration.
        mark_alive_nodes : bool, optional (default=False)
            If True, nodes, which are not alive will have additional special sign near their name.
        normalization : string, optional (default=None)
            Defines method of graph edges normalization.
            Available options are: 'diet', 'log', 'donor_control',
            'predator_control', 'mixed_control', 'linear' and 'TST'.
        no_flows_to_detritus : bool, optional (default=False)
            If True, fLows to detritus will be excluded from the results.

        Returns
        -------
        subgraph : networkx.SubGraph
            A read-only restricted view of networkx.DiGraph.
        '''
        exclude_nodes = [] if boundary else ['Import', 'Export', 'Respiration']

        exclude_edges = []
        if no_flows_to_detritus:
            not_alive_nodes = self.node_df[~self.node_df.IsAlive].index.values
            exclude_edges = [edge for edge in self._graph.edges() if edge[1] in not_alive_nodes]

        g = nx.restricted_view(self._graph.copy(), exclude_nodes, exclude_edges)
        if mark_alive_nodes:
            g = nx.relabel_nodes(g, fw.is_alive_mapping(self))
        g = normalization_factory(g, norm_type=normalization)
        return g

    def get_flows(self, boundary=False, mark_alive_nodes=False, normalization=None,
                  no_flows_to_detritus=False):
        '''Returns a list of all flows within foodweb.

        Parameters
        ----------
        boundary : bool, optional (default=False)
            If True, boundary flows will be added to the graph.
            Boundary flows are: Import, Export, and Repiration.
        mark_alive_nodes : bool, optional (default=False)
            If True, nodes, which are not alive will have additional special sign near their name.
        normalization : string, optional (default=None)
            Defines method of graph edges normalization.
            Available options are: 'diet', 'log', 'donor_control',
            'predator_control', 'mixed_control', 'linear' and 'tst'.
        no_flows_to_detritus : bool, optional (default=False)
            If True, fLows to detritus will be excluded from the results.

        Returns
        -------
        flows : list of tuples
            List of edges in graph's representation of a foodweb,
            each tuple is in a form of (from, to, weight).

        '''
        return (self.get_graph(boundary, mark_alive_nodes, normalization, no_flows_to_detritus)
                .edges(data=True))

    def get_flow_matrix(self, boundary=False, to_alive_only=False):
        '''Returns the flow (adjacency) matrix.

        Parameters
        ----------
        boundary : bool, optional (default=False)
            If True, boundary flows will be added to the graph.
            Boundary flows are: Import, Export, and Repiration.
        to_alive_only : bool, optional (default=False)
            If True, flow_matrix will include only flows to alive nodes
            (flows to not alive nodes will be 0)

        Returns
        -------
        flows_matrix : pd.DataFrame
            Rows/columns are species, each row/column intersection represents flow
            from ith to jth node.
        '''
        flow_matrix = self.flow_matrix.copy()

        if to_alive_only:
            flow_matrix.transpose()[~self.node_df['IsAlive']] = 0.0

        if not boundary:
            return flow_matrix

        flow_matrix_with_boundary = self.flow_matrix.copy()
        flow_matrix_with_boundary.loc['Import'] = self.node_df.Import.to_dict()
        flow_matrix_with_boundary.loc['Export'] = self.node_df.Export.to_dict()
        flow_matrix_with_boundary.loc['Respiration'] = self.node_df.Respiration.to_dict()
        return (
            flow_matrix_with_boundary
            .join(self.node_df.Import)
            .join(self.node_df.Export)
            .join(self.node_df.Respiration)
            .fillna(0.0))

    def get_links_number(self):
        '''Returns the number of nonzero flows.
        '''
        return self.get_graph(False).number_of_edges()

    def get_flow_sum(self):
        '''Returns the sum of all flows.
        '''
        return self.get_flow_matrix(boundary=True).sum()

    def get_norm_node_prop(self):
        num_node_prop = self.node_df[["Biomass", "Import", "Export", "Respiration"]]
        return num_node_prop.div(num_node_prop.sum(axis=0), axis=1)

    def __str__(self):
        return f'''
                {self.title}\n
                {self.node_df["Biomass"]}\n
                The internal flows matrix: a_ij=flow from i to j\n
                {self.flow_matrix}\n'
                {self.node_df["Import"]}\n
                {self.node_df["Export"]}\n
                {self.node_df["Respiration"]}\n
                {self.node_df["TrophicLevel"]}\n
                '''

    def get_outflows_to_living(self):
        # node's system outflows to living
        # TODO doc
        return self.flow_matrix[self.node_df[self.node_df.IsAlive].index].sum(axis='columns')
