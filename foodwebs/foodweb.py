import networkx as nx

import foodwebs as fw
from .normalization import normalization_factory


__all__ = [
    'FoodWeb'
]


class FoodWeb():
    '''
    Class defining a food web of an ecosystem with given stock biomasses
    and flows between species (compartments)"
    '''

    def __init__(self, title, node_df, flow_matrix):
        '''
        Constructor from given dataframes

        Parameters:
        title - name of the foodweb
        node_df - dataframe with columns:
            "Name", "IsLiving", "Biomass", "Import", "Export", "TrophicLevel", "Respiration"
        flow_matrix - dataframe of system flows within the ecosystem TODO columns?
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
        '''
        Initialized networkx graph using adjacency matrix.

        Returns:
        networkx.DiGraph
        '''
        graph = nx.from_pandas_adjacency(self.get_flow_matrix(boundary=True),  create_using=nx.DiGraph)
        nx.set_node_attributes(graph, self.node_df.to_dict(orient='index'))

        exclude_edges = []
        for n in self.node_df.index.values:
            exclude_edges.append((n, 'Import'))
            exclude_edges.append(('Export', n))
            exclude_edges.append(('Respiration', n))
        graph.remove_edges_from(exclude_edges)
        return graph

    def get_graph(self, boundary=False, mark_alive_nodes=False, normalization=None):
        '''
        Allows access to networkx representation of foodweb.

        Parameters:
        boundary - add boundary flows (Import, Export, Repiration) to the graph
        mark_alive_nodes - nodes, which are not alive will have additional X mark near their name
        normalization - additional normalization method to apply on graph edges.
            Avaiable options are: diet, log, biomass, tst

        Returns:
        view of networkx.DiGraph
        '''
        exclude_nodes = [] if boundary else ['Import', 'Export', 'Respiration']

        g = nx.restricted_view(self._graph.copy(), exclude_nodes, [])
        if mark_alive_nodes:
            g = nx.relabel_nodes(g, fw.is_alive_mapping(self))
        g = normalization_factory(g, norm_type=normalization)
        return g

    def get_flows(self, boundary=False, mark_alive_nodes=False, normalization=None):
        '''
        Returns a dataframe of all internal flows in a form [from, to, weight]

        Parameters:
        boundary - add boundary flows (Import, Export, Repiration) to the graph
        mark_alive_nodes - nodes, which are not alive will have additional X mark near their name
        normalization - additional normalization method to apply on graph edges.
            Avaiable options are: diet, log, biomass, tst

        Returns:
        list of tuples

        '''
        return self.get_graph(boundary, mark_alive_nodes, normalization).edges(data=True)

    def get_flow_matrix(self, boundary=False):
        '''
        Returns the flow (adjacency) matrix.

        Parameters:
        boundary - add boundary flows (Import, Export, Repiration) to the matrix

        Returns:
        pd.DataFrame
        '''
        if not boundary:
            return self.flow_matrix

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
        '''
        Returns the number of nonzero system links
        '''
        return self.get_graph(False).number_of_edges()

    def get_flow_sum(self):
        '''
        Returns the sum of ALL flows
        '''
        return self.get_flow_matrix(boundary=True).sum()

    def get_norm_node_prop(self):
        num_node_prop = self.node_df[["Biomass", "Import", "Export", "Respiration"]]
        return(num_node_prop.div(num_node_prop.sum(axis=0), axis=1))

    def __str__(self):
        '''
        Overloading print operator
        '''
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
