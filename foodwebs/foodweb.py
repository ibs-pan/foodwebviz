import numpy as np
import pandas as pd


def count(df, cond):
    # count dataframe 'df' elements for which a condition 'cond' holds
    return df.apply(cond).values.sum()


def find_edges(df, ifNames):
    """Finds the edges in the square adjacency matrix, using
    vectorized operations. Returns a list of pairs of tuples
    that represent the edges."""
    values = df.values
    n_rows, n_columns = values.shape

    indices = np.arange(n_rows * n_columns)
    values = values.flatten()
    # indices = indices[values > 0.0]
    # A value of 1 means that the edge exists

    # Create two arrays `rows` and `columns` such that for an edge i,
    # (rows[i], columns[i]) is its coordinate in the df
    rows = [int(x) for x in indices / n_columns]
    columns = [int(y) for y in indices % n_columns]
    # Convert the coordinates to actual names
    row_names = df.index[rows]
    column_names = df.columns[columns]
    # python numbers indices from 0, we want from 1 (required by Ecomodels, which throw an exception otherwise
    rows = [x+1 for x in rows]
    columns = [y+1 for y in columns]

    if ifNames:
        l = list(zip(row_names, column_names, values))
    else:
        l = list(zip(rows, columns, values))
    return [x for x in l if x[2] != 0.0]


class FoodWeb:
    '''Class defining a food web of an ecosystem with given stock biomasses
    and flows between species (compartments)"
    '''
    def __init__(self, title='Empty title', nodeDF=None, flowMatrix=None, n=None, n_living=None):
        '''Constructor from given dataframes'''
        self.title = title
        # a dataframe with columns:
        # "Name", "IsLiving", "Biomass", "Import", "Export", "TrophicLevel", "Respiration"
        self.nodeDF = nodeDF
        # a dataframe of system flows within the ecosystem
        self.flowMatrix = flowMatrix
        # number of nodes
        self.n = n
        # number of living nodes
        self.n_living = n_living

        # a dict of names: numbers
        self.NrName = None
        # a dict of node numbers: names
        self.NameNr = None

        # we can calculate trophic levels
        if len(flowMatrix) > 1:
            self.nodeDF['TrophicLevel'] = self.calcTrophicLevels()

        self.nodeDF = self.nodeDF.set_index(nodeDF["Names"])

    def writeXLS(self, filename):
        '''Save the FoodWeb as an XLS file - spreadsheets.'''
        print(f'Saving FoodWeb with title {self.title}')
        writer = pd.ExcelWriter(filename)
        pd.DataFrame([self.title]).to_excel(writer, sheet_name="Title")
        self.nodeDF.to_excel(writer, sheet_name="Node properties")
        self.flowMatrix.to_excel(writer, sheet_name="Internal flows")
        writer.save()

    def getFlows(self, ifNames=False):
        '''Function returns a long dataframe of all internal flows'''
        return find_edges(self.flowMatrix, ifNames)

    def getExternalFlows(self):
        external_flows = []
        for i, row in self.nodeDF.iterrows():
            for col in ['Import', 'Export', 'Respiration']:
                if row[col] != 0:
                    external_flows.append((row.Names, col, row[col]))
        return external_flows

    def getNormInternFlows(self):
        '''Function returning a list of internal flows normalized to TST'''
        flows = find_edges(self.flowMatrix, False)
        TST = sum(link[2] for link in flows)
        return [x[2] / TST for x in flows]

    def getNormNodeProp(self):
        numNodeProp = self.nodeDF.loc[:, [
            "Biomass", "Import", "Export", "Respiration"]]
        return(numNodeProp.div(numNodeProp.sum(axis=0), axis=1))

    def genFlow(self, imports, exports):
        '''Return the generalised flow matrix - here working on Dataframes
        with named rows lead to reshuffling'''
        # adding imports as the last row
        genMat = np.concatenate(
            (self.nodeDF.Import.values, imports.values.transpose()), axis=0)
        # numpy array of exports with 0 at the end as exports x
        # imports field = flow from environment to the environment
        exp = np.expand_dims(
            np.append(exports.values, [0]), axis=0).transpose()

        # returns internal flows with exports added as the last column
        return np.concatenate((genMat, exp), axis=1)

    def getLinksNr(self):
        '''Returns the number of nonzero system links'''
        return count(self.flowMatrix, lambda l: l > 0.0)

    def getFlowMatWithBoundary(self):
        '''Returns the flow matrix including the boundary flows as the last row and column'''
        return self.genFlow(self.flowMatrix,
                            self.nodeDF.Import,
                            self.nodeDF.Export + self.nodeDF.Respiration.values)

    def setFlows(self, genFlowMatrix):
        '''Set the internal attributes from a generalized flow matrix'''
        # exports sit in the last column, but we exclude the last row,
        # cornodeDF.loc[:,"Respiration"]onding to the environment
        self.nodeDF['Export'] = pd.Series(
            genFlowMatrix[:len(genFlowMatrix)-1,
                          len(genFlowMatrix[0])-1])
        self.nodeDF['Export'].index = np.arange(
            1, len(self.nodeDF['Export']) + 1)

        # imports sit in the last row
        self.nodeDF['Import'] = pd.Series(genFlowMatrix[len(
            genFlowMatrix)-1, :len(genFlowMatrix)-1])
        self.nodeDF['Import'].index = np.arange(
            1, len(self.nodeDF['Import']) + 1)

        self.flowMatrix = pd.DataFrame(
            genFlowMatrix[:len(genFlowMatrix)-1, :len(genFlowMatrix[0])-1])

        # nodeDF.loc[:,"Respiration"]irations are not included so far in the null model!!!
        # self.nodeDF.loc[:,"Respiration"]=pd.Series(np.zeros(len(genFlowMatrix)-1))
        # self.nodeDF.loc[:,"Respiration"].index = np.arange(1, len(self.nodeDF.loc[:,"Respiration"]) + 1)

    def getFlowSum(self):
        "Returns the sum of ALL flows"
        return self.getFlowMatWithBoundary().sum()

    def wrsep(self, filename):  # add the separating -1 to the SCOR file
        with open(filename, 'a') as f:
            f.write('-1 \n')

    def idNr(self, attr):
        # rename the index so that the result is indexed by numbers instead of names of species/compartments
        # check if the attribute is not already just integers
        names = attr.index
        if str(names[0]).isdigit():
            return(attr)
        else:
            return(attr.rename(index=self.NameNr))

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
        A = np.zeros([dataSize, dataSize])
        inflow = np.zeros(dataSize)
        isFixedToOne = np.zeros(dataSize, dtype=bool)
        dataTrophicLevel = np.zeros(dataSize)

        n = 0  # counting the nodes with TL fixed to 1
        for i in range(0, dataSize):
            isFixedToOne[i] = False
            dataTrophicLevel[i] = 0.0
            for j in range(0, dataSize):
                # sum of all incoming system flows to the compartment i
                inflow[i] += self.flowMatrix.iloc[j, i]
                # the diagonal has the sum of all incoming system flows to the compartment i,
                # except flow from i to i
                A[i][i] += self.flowMatrix.iloc[j, i]
                if i != j:
                    A[i][j] = -self.flowMatrix.iloc[j, i]

            # choose the nodes that have TL=1: non-living and primary producers == inflows equal to zero
            if (inflow[i] <= 0.0 or i >= int(self.n_living)):
                isFixedToOne[i] = True
                dataTrophicLevel[i] = 1.0
                n += 1

        if (n != 0):
            # update the equation due to the prescribed trophic level 1 - reduce the dimension of the matrix
            B_tmp = np.zeros(dataSize - n)
            A_tmp = np.zeros([dataSize - n, dataSize - n])
            TL_tmp = np.zeros(dataSize - n)
            tmp_i = 0
            for i in range(0, dataSize):
                if not isFixedToOne[i]:
                    B_tmp[tmp_i] = inflow[i]
                    tmp_j = 0
                    for j in range(0, dataSize):
                        if (isFixedToOne[j]):  # means also i!=j
                            # moving the contribution to the constant part,
                            # + flow(j->i)*1 to both sides of the equation
                            B_tmp[tmp_i] -= A[i][j]
                        else:
                            A_tmp[tmp_i][tmp_j] = A[i][j]
                            tmp_j += 1
                    tmp_i += 1

            for i in range(0, dataSize-n):
                TL_tmp[i] = 0.0
                Ainverse = np.linalg.pinv(A_tmp)
                for j in range(0, dataSize-n):

                    TL_tmp[i] += Ainverse[i][j] * B_tmp[j]

            k = 0
            for i in range(0, dataSize):
                if (not isFixedToOne[i]):
                    dataTrophicLevel[i] = TL_tmp[k]
                    k += 1
        else:
            try:
                A1 = np.linalg.pinv(A)
            except:
                # negative trophic levels signify some problems
                # print("The matrix A1 which inverse was seeked but does not exist:")
                print(A1)
        return dataTrophicLevel

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
