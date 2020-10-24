import pandas as pd
from .foodweb import FoodWeb


def read_from_SCOR(scor_path):
    '''
    Reads a TXT file in the SCOR format and returns a FoodWeb object

      SCOR file:
      -------------------------------------------------------------
      title
      #of all compartments #of living compartments
      1st compartment name
      2nd compartment name
      ...
      biomasses (stock)  // line -> vector element
      -1
      imports (B^in)
      -1
      exports (B^out)
      -1
      respiration (R)
      -1
      flows ((Victoria's S)^T)  // line -> matrix element
      -1 -1
      ----------------------------------------------------------
    '''
    with open(scor_path, 'r', encoding='utf-8') as f:
        title = f.readline().strip()
        size = f.readline().split()
        assert len(size) == 2
        n = int(size[0])
        n_living = int(size[1])

        lines = [x.strip() for x in f.readlines()]

        net = pd.DataFrame(index=range(1, n+1))
        net['Names'] = lines[:n]
        net['IsAlive'] = [i < n_living for i in range(n)]

        for i, col in enumerate(['Biomass', 'Import', 'Export', 'Respiration']):
            net[col] = [float(x.split(' ')[1])
                        for x in lines[(i + 1) * n + i: (i + 2) * n + i]]

        flow_matrix = pd.DataFrame(index=range(1, n+1), columns=range(1, n+1))
        for line in [x.split(' ') for x in lines[(i + 2) * n + i + 1: -1]]:
            flow_matrix.at[int(line[0]), int(line[1])] = float(line[2])
        flow_matrix = flow_matrix.fillna(0.0)
        flow_matrix.index = net.Names
        flow_matrix.columns = net.Names
        return FoodWeb(title=title, flow_matrix=flow_matrix, node_df=net)
