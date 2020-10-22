import pandas as pd
from .foodweb import FoodWeb


def readFW_SCOR(scor_path):
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

        flowMatrix = pd.DataFrame(index=range(1, n+1), columns=range(1, n+1))
        for line in [x.split(' ') for x in lines[(i + 2) * n + i + 1: -1]]:
            flowMatrix.at[int(line[0]), int(line[1])] = float(line[2])
        flowMatrix = flowMatrix.fillna(0.0)
        flowMatrix.index = net.Names
        flowMatrix.columns = net.Names

        return FoodWeb(title=title, flowMatrix=flowMatrix, nodeDF=net)


def writeXLS(fw, filename):
    "Save the FoodWeb as an XLS file - spreadsheets."
    print(f'Saving FoodWeb with title {fw.title}')
    writer = pd.ExcelWriter(filename,  encoding='utf-8')
    pd.DataFrame(index=[filename.strip('.xls').split('/')[-1]],
                 data=fw.title,
                 columns=["Title"]).to_excel(writer, sheet_name="Title")
    fw.nodeDF.to_excel(writer, sheet_name="Node properties")
    fw.flowMatrix.to_excel(writer, sheet_name="Internal flows")
    writer.save()
