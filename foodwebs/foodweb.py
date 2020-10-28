import numpy as np
import pandas as pd
import networkx as nx

from .normalization import flows_normalization
from .visualization import show_heatmap


class FoodWeb():
    '''Class defining a food web of an ecosystem with given stock biomasses
    and flows between species (compartments)"
    '''
    def __init__(self, title, node_df, flow_matrix):
        '''Constructor from given dataframes'''
        self.title = title

        # dataframe with columns:
        # "Name", "IsLiving", "Biomass", "Import", "Export", "TrophicLevel", "Respiration"
        self.node_df = node_df
        self.node_df = self.node_df.set_index("Names")

        # dataframe of system flows within the ecosystem
        self.flow_matrix = flow_matrix

        self.n = len(self.node_df)
        self.n_living = len(self.node_df[self.node_df.IsAlive])

        # calculate trophic levels
        if len(flow_matrix) > 1:
            self.node_df['TrophicLevel'] = self._calculate_trophic_levels()
        self._graph = self._init_graph()

    def _init_graph(self):
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
        exclude_nodes = [] if boundary else ['Import', 'Export', 'Respiration']

        g = nx.restricted_view(self._graph.copy(), exclude_nodes, [])
        if mark_alive_nodes:
            g = nx.relabel_nodes(g, self._is_alive_mapping())
        g = flows_normalization(g, norm_type=normalization)
        return g

    def get_flows(self, boundary=False, mark_alive_nodes=False, normalization=None):
        '''Function returns a long dataframe of all internal flows'''
        return self.get_graph(boundary, mark_alive_nodes, normalization).edges(data=True)

    def get_flow_matrix(self, boundary=False):
        '''Returns the flow matrix including the boundary flows as the last row and column'''
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
        '''Returns the number of nonzero system links'''
        return self.get_graph(False).number_of_edges()

    def get_flow_sum(self):
        "Returns the sum of ALL flows"
        return self.get_flow_matrix(boundary=True).sum()

    def _is_alive_mapping(self):
        '''
        Creates dictionary which special X character to names, which are not alive
        '''
        return {name: f'\u2717 {name}' for name in self.node_df[~self.node_df.IsAlive].index.values}

    def get_norm_node_prop(self):
        num_node_prop = self.node_df[["Biomass", "Import", "Export", "Respiration"]]
        return(num_node_prop.div(num_node_prop.sum(axis=0), axis=1))

    def _calculate_trophic_levels(self):
        '''function calculating the trophic levels of nodes from the recursive relation'''
        data_size = len(self.flow_matrix)

        # sum of all incoming system flows to the compartment i
        inflow_pd = pd.DataFrame(self.flow_matrix.sum(axis=0), columns=['inflow'])

        # the diagonal has the sum of all incoming system flows to the compartment i,
        # except flow from i to i
        A = self.flow_matrix.values.transpose() * -1
        np.fill_diagonal(A, inflow_pd.values)

        inflow_pd['is_fixed_to_one'] = (inflow_pd.inflow <= 0.0) | (np.arange(data_size) >= self.n_living)
        inflow_pd['data_trophic_level'] = inflow_pd.is_fixed_to_one.astype(float)
        inflow_pd = inflow_pd.reset_index()

        # counting the nodes with TL fixed to 1
        if (sum(inflow_pd.is_fixed_to_one) != 0):
            not_one = inflow_pd[~inflow_pd.is_fixed_to_one].index.values
            one = inflow_pd[inflow_pd.is_fixed_to_one].index.values

            # update the equation due to the prescribed trophic level 1 - reduce the dimension of the matrix
            A_tmp = A[np.ix_(not_one, not_one)]

            B_tmp = inflow_pd[~inflow_pd.is_fixed_to_one].inflow.values
            B_tmp -= np.sum(A[np.ix_(not_one, one)], axis=1)

            A_inverse = np.linalg.pinv(A_tmp)
            A_inverse = np.multiply(A_inverse, B_tmp)
            inflow_pd.loc[~inflow_pd['is_fixed_to_one'], 'data_trophic_level'] = np.sum(A_inverse, axis=1)
        else:
            # fails with negative trophic levels = some problems
            np.linalg.pinv(A)
        return inflow_pd.data_trophic_level.values

    def show_heatmap(self, normalization=None, show_trophic_layer=True, add_external_flows=False, switch_axes=False):
        show_heatmap(self,
                     normalization=normalization,
                     show_trophic_layer=show_trophic_layer,
                     add_external_flows=add_external_flows)

    def writeXLS(self, filename):
        '''Save the FoodWeb as an XLS file - spreadsheets.'''
        print(f'Saving FoodWeb with title {self.title}')
        writer = pd.ExcelWriter(filename)
        pd.DataFrame([self.title]).to_excel(writer, sheet_name="Title")
        self.node_df.to_excel(writer, sheet_name="Node properties")
        self.flow_matrix.to_excel(writer, sheet_name="Internal flows")
        writer.save()

    def write_SCOR(self, filename):
        ''' Write food web to a SCOR file'''
        def write_col(node_df, f, col):
            node_df[col].to_csv(f, header=None, sep=' ', mode='a')
            f.write('-1 \n')

        with open(filename, 'w') as f:
            # save the title of the network
            f.write(f'{self.title} \n')
            # number of compartments
            f.write(f'{self.n} {self.n_living} \n')
            # names of the species/compartments
            f.writelines([f'{x}\n' for x in self.node_df.index])

            node_df = self.node_df.reset_index().copy()
            node_df.index = node_df.index + 1
            for col in ['Biomass', 'Import', 'Export', 'Respiration']:
                write_col(node_df, f, col)

            n_map = node_df.reset_index().set_index('Names')[['index']].to_dict()['index']
            f.writelines(
                [f'{n_map[edge[0]]} {n_map[edge[1]]} {edge[2]["weight"]}\n' for edge in self.get_flows()])
            f.write('-1 \n')

    def __str__(self):
        '''Overloading print operator'''
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
