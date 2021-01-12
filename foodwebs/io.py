"""Functions to read Foodweb objects from other formats.

Examples
--------

Create a foodweb from a SCOR file
>>> food_web = read_from_SCOR(file_path)
"""


import pandas as pd
from .foodweb import FoodWeb


__all__ = [
    'read_from_SCOR',
    'write_to_SCOR',
    'write_to_XLS'
]


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

    Parameters:
    scor_path - file path

    Returns:
    foodweb.FoodWeb
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


def write_to_SCOR(food_web, scor_path):
    '''
    Write food web to a SCOR file.
    '''
    def write_col(node_df, f, col):
        node_df[col].to_csv(f, header=None, sep=' ', mode='a')
        f.write('-1 \n')

    with open(scor_path, 'w') as f:
        f.write(f'{food_web.title} \n')
        f.write(f'{food_web.n} {food_web.n_living} \n')
        f.writelines([f'{x}\n' for x in food_web.node_df.index])

        node_df = food_web.node_df.reset_index().copy()
        node_df.index = node_df.index + 1
        for col in ['Biomass', 'Import', 'Export', 'Respiration']:
            write_col(node_df, f, col)

        n_map = node_df.reset_index().set_index('Names')[['index']].to_dict()['index']
        f.writelines(
            [f'{n_map[edge[0]]} {n_map[edge[1]]} {edge[2]["weight"]}\n' for edge in food_web.get_flows()])
        f.write('-1 \n')


def write_to_XLS(food_web, filename):
    '''
    Save the FoodWeb as an XLS file - spreadsheets.
    '''
    print(f'Saving FoodWeb with title {food_web.title}')

    writer = pd.ExcelWriter(filename)
    pd.DataFrame([food_web.title]).to_excel(writer, sheet_name="Title")
    food_web.node_df.to_excel(writer, sheet_name="Node properties")
    food_web.flow_matrix.to_excel(writer, sheet_name="Internal flows")
    writer.save()
