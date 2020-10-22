import numpy as np
import pandas as pd
import networkx as nx

from .normalization import flows_normalization


class FoodWeb:
    '''Class defining a food web of an ecosystem with given stock biomasses
    and flows between species (compartments)"
    '''

    def __init__(self, title, nodeDF, flowMatrix):
        '''Constructor from given dataframes'''
        self.title = title

        # a dataframe with columns:
        # "Name", "IsLiving", "Biomass", "Import", "Export", "TrophicLevel", "Respiration"
        self.nodeDF = nodeDF
        self.nodeDF = self.nodeDF.set_index("Names")

        # a dataframe of system flows within the ecosystem
        self.flowMatrix = flowMatrix

        # number of nodes
        self.n = len(self.nodeDF)

        # number of living nodes
        self.n_living = len(self.nodeDF[self.nodeDF.IsAlive])

        # we can calculate trophic levels
        if len(flowMatrix) > 1:
            self.nodeDF['TrophicLevel'] = self.calcTrophicLevels()

        self._graph = self._init_graph()

    def _init_graph(self):
        graph = nx.from_pandas_adjacency(self.getFlowMatWithBoundary(),  create_using=nx.DiGraph)
        nx.set_node_attributes(graph, self.nodeDF.to_dict(orient='index'))

        exclude_edges = []
        for n in self.nodeDF.index.values:
            exclude_edges.append((n, 'Import'))
            exclude_edges.append(('Export', n))
            exclude_edges.append(('Respiration', n))
        graph.remove_edges_from(exclude_edges)
        return graph

    def getGraph(self, boundary=False, mark_alive_nodes=False, normalization=None):
        exclude_nodes = [] if boundary else ['Import', 'Export', 'Respiration']

        g = nx.restricted_view(self._graph.copy(), exclude_nodes, [])
        if mark_alive_nodes:
            g = nx.relabel_nodes(g, self._is_alive_mapping())
        g = flows_normalization(g, norm_type=normalization)
        return g

    def getFlows(self, boundary=False, mark_alive_nodes=False, normalization=None):
        '''Function returns a long dataframe of all internal flows'''
        return self.getGraph(boundary, mark_alive_nodes, normalization).edges(data=True)

    def getFlowMatWithBoundary(self):
        '''Returns the flow matrix including the boundary flows as the last row and column'''
        flowMatrixWithBoundary = self.flowMatrix.copy()
        flowMatrixWithBoundary.loc['Import'] = self.nodeDF.Import.to_dict()
        flowMatrixWithBoundary.loc['Export'] = self.nodeDF.Export.to_dict()
        flowMatrixWithBoundary.loc['Respiration'] = self.nodeDF.Respiration.to_dict()
        return (
            flowMatrixWithBoundary
            .join(self.nodeDF.Import)
            .join(self.nodeDF.Export)
            .join(self.nodeDF.Respiration)
            .fillna(0.0)
        )

    def getLinksNr(self):
        '''Returns the number of nonzero system links'''
        return self.getGraph(False).number_of_edges()

    def getFlowSum(self):
        "Returns the sum of ALL flows"
        return self.getFlowMatWithBoundary().sum()

    def _is_alive_mapping(self):
        '''
        Creates dictionary which special X character to names, which are not alive
        '''
        return {name: f'\u2717 {name}' for name in self.nodeDF[~self.nodeDF.IsAlive].index.values}

    def writeXLS(self, filename):
        '''Save the FoodWeb as an XLS file - spreadsheets.'''
        print(f'Saving FoodWeb with title {self.title}')
        writer = pd.ExcelWriter(filename)

        # save title
        pd.DataFrame([self.title]).to_excel(writer, sheet_name="Title")

        # save nodes DataFrame
        self.nodeDF.to_excel(writer, sheet_name="Node properties")

        # save flow matrix
        self.flowMatrix.to_excel(writer, sheet_name="Internal flows")
        writer.save()

    def getNormNodeProp(self):
        numNodeProp = self.nodeDF[["Biomass", "Import", "Export", "Respiration"]]
        return(numNodeProp.div(numNodeProp.sum(axis=0), axis=1))

    def wrsep(self, filename):  # add the separating -1 to the SCOR file
        with open(filename, 'a') as f:
            f.write('-1 \n')

    def write_SCOR(self, filename):
        # function writing the current food web ('self') to a SCOR file
        with open(filename, 'w') as f:
            f.write(self.title + ' \n')  # save the title of the network
            # number of compartments
            f.write(str(self.n)+' '+str(self.n_living)+' \n')
            for s in self.names:  # names of the species/compartments
                f.write(str(s) + ' \n')

        self.idNr(self.nodeDF.loc[:, "Biomass"]).to_csv(
            filename, header=None,  sep=' ', mode='a')  # save the nodeDF.loc[:,"Biomass"]
        self.wrsep(filename)
        infl = self.idNr(self.nodeDF.loc[:, "Import"])  # save the imports
        infl.to_csv(filename, header=None, sep=' ', mode='a')
        self.wrsep(filename)
        outfl = self.idNr(self.nodeDF.loc[:, "Export"])  # save the exports
        outfl.to_csv(filename, header=None, sep=' ', mode='a')
        self.wrsep(filename)
        self.idNr(self.nodeDF.loc[:, "Respiration"]).to_csv(
            filename, header=None,  sep=' ', mode='a')
        self.wrsep(filename)

        with open(filename, 'a') as f:
            # write the internal flows as edges list
            edgList = find_edges(self.flowMatrix, False)

            for edge in edgList:
                for x in edge:
                    f.write(str(x)+' ')
                f.write('\n')
        self.wrsep(filename)

    def calcTrophicLevels(self):
        '''function calculating the trophic levels of nodes from the recursive relation'''
        dataSize = len(self.flowMatrix)

        # sum of all incoming system flows to the compartment i
        inflow_pd = pd.DataFrame(self.flowMatrix.sum(axis=0), columns=['inflow'])

        # the diagonal has the sum of all incoming system flows to the compartment i,
        # except flow from i to i
        A = self.flowMatrix.values.transpose() * -1
        np.fill_diagonal(A, inflow_pd.values)

        inflow_pd['isFixedToOne'] = (inflow_pd.inflow <= 0.0) | (np.arange(dataSize) >= self.n_living)
        inflow_pd['dataTrophicLevel'] = inflow_pd.isFixedToOne.astype(float)
        inflow_pd = inflow_pd.reset_index()

        # counting the nodes with TL fixed to 1
        if (sum(inflow_pd.isFixedToOne) != 0):
            not_one = inflow_pd[~inflow_pd.isFixedToOne].index.values
            one = inflow_pd[inflow_pd.isFixedToOne].index.values

            # update the equation due to the prescribed trophic level 1 - reduce the dimension of the matrix
            A_tmp = A[np.ix_(not_one, not_one)]

            B_tmp = inflow_pd[~inflow_pd.isFixedToOne].inflow.values
            B_tmp -= np.sum(A[np.ix_(not_one, one)], axis=1)

            Ainverse = np.linalg.pinv(A_tmp)
            Ainverse = np.multiply(Ainverse, B_tmp)
            inflow_pd.loc[~inflow_pd['isFixedToOne'], 'dataTrophicLevel'] = np.sum(Ainverse, axis=1)
        else:
            # fails with negative trophic levels = some problems
            np.linalg.pinv(A)
        return inflow_pd.dataTrophicLevel.values

    def __str__(self):
        '''Overloading print operator'''
        return f'''
                {self.title}\n
                {self.nodeDF.loc[:,"Biomass"]}\n
                {self.nodeDF.loc[:,"Biomass"]}\n
                The internal flows matrix: a_ij=flow from i to j\n
                {self.flowMatrix}\n'
                {self.nodeDF.loc[:,"Import"]}\n
                {self.nodeDF.loc[:,"Export"]}\n
                {self.nodeDF.loc[:,"Respiration"]}\n
                {self.nodeDF.loc[:,"TrophicLevel"]}\n
                '''
