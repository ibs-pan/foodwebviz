'''Functions to read Foodweb objects from other formats.

Examples
--------

Create a foodweb from a SCOR file
>>> food_web = read_from_SCOR(file_path)
'''

import numpy as np
import pandas as pd
import foodwebviz as fw


__all__ = [
    'write_to_SCOR',
    'write_to_XLS',
    'write_to_CSV',
    'read_from_SCOR',
    'read_from_XLS',
    'read_from_CSV'
]


def read_from_SCOR(scor_path):
    '''Reads a TXT file in the SCOR format and returns a FoodWeb object

    SCOR file has the following format:
    -------------------------------------------------------------
    title
    #of all compartments #of living compartments <-- size
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
    -1
    ----------------------------------------------------------

    Example:
    ----------------------------------------------------------
    example_foodweb_1
    2 1
    A
    B
    1 0.00303
    2 0.05
    -1
    1 0.0018666315
    2 0.0
    -1 
    1 3.35565e-07
    2 0.0001
    -1 
    1 0.09925068
    2 1.45600009
    -1 
    1 2 0.002519108
    -1
    ----------------------------------------------------------

    Parameters
    ----------
    scor_path : string
        Path to the foodweb in SCOR format.


    Returns
    -------
    foodweb : foodwebs.Foodweb
    '''
    with open(scor_path, 'r', encoding='utf-8') as f:
        print(f'Reading file: {scor_path}')
        title = f.readline().strip()
        size = f.readline().split()

        # check that size line has two values
        if len(size) != 2:
            raise Exception('Invalid SCOR file format.')

        n, n_living = int(size[0]), int(size[1])
        if n_living > n:
            raise Exception('Invalid input. The number of living species \
                             has to be smaller than the number of all nodes.')

        if n <= 0 or n_living <= 0:
            raise Exception('Number of nodes and number of living nodes have to be positive integers.')

        lines = [x.strip() for x in f.readlines()]

        net = pd.DataFrame(index=range(1, n+1))
        net['Names'] = lines[:n]
        net['IsAlive'] = [i < n_living for i in range(n)]

        for i, col in enumerate(['Biomass', 'Import', 'Export', 'Respiration']):
            # each section should end with -1
            if lines[(i + 1) * n + i + n] != '-1':
                raise Exception(f'Invalid SCOR file format. {col} section could be wrong, \
                                  the separator -1 could be in a wrong place, names list \
                                  could have wrong length.')

            net[col] = [float(x.split(' ')[1])
                        for x in lines[(i + 1) * n + i: (i + 2) * n + i]]

        flow_matrix = pd.DataFrame(index=range(1, n+1), columns=range(1, n+1))
        for line in [x.split(' ') for x in lines[(i + 2) * n + i + 1: -1]]:
            flow_matrix.at[int(line[0]), int(line[1])] = float(line[2])
        flow_matrix = flow_matrix.fillna(0.0)
        flow_matrix.index = net.Names
        flow_matrix.columns = net.Names
        return fw.FoodWeb(title=title, flow_matrix=flow_matrix, node_df=net)


def write_to_SCOR(food_web, scor_path):
    '''Write foodweb to a SCOR file.

    Parameters
    ----------
    foodweb : foodwebs.FoodWeb
        Object to save.
    scor_path: string
        Destination path.

    See Also
    --------
    io.read_from_SCOR
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
    '''Write foodweb as an XLS (spreadsheet) file.

    Parameters
    ----------
    foodweb : foodwebs.FoodWeb
        Object to save.
    filename: string
        Destination path.
    '''
    writer = pd.ExcelWriter(filename)
    pd.DataFrame([food_web.title]).to_excel(writer, sheet_name="Title")
    food_web.node_df.to_excel(writer, sheet_name="Node properties")
    food_web.flow_matrix.to_excel(writer, sheet_name="Internal flows")
    writer.save()


def read_from_XLS(filename):
    '''Read foodweb from an XLS (spreadsheet) file.

    Parameters
    ----------
    foodweb : foodwebs.FoodWeb
        Object to save.
    filename: string
        Destination path.
    '''
    title = pd.read_excel(filename, sheet_name='Title')
    node_df = pd.read_excel(filename, sheet_name='Node properties',
                            # will fail if any of those columns is missing
                            usecols=['Names', 'IsAlive', 'Biomass', 'Import', 'Export', 'Respiration'],
                            dtype={'Names': str,
                                   'IsAlive': bool,
                                   'Biomass': np.float64,
                                   'Import': np.float64,
                                   'Export': np.float64,
                                   'Respiration': np.float64})

    flow_matrix = pd.read_excel(filename, sheet_name='Internal flows')
    if not np.array_equal(flow_matrix.columns.values[1:], flow_matrix.Names.values):
        raise Exception('Flow matrix (Internal flows sheet) should have exactly same rows as columns.')
    if (flow_matrix < 0).any().any():
        raise Exception('Flow matrix contains negative values.')

    names = flow_matrix.Names
    flow_matrix.drop('Names', inplace=True, axis=1)
    flow_matrix.index = names
    flow_matrix.columns = names
    return fw.FoodWeb(title=title.values[0][1], node_df=node_df, flow_matrix=flow_matrix)


def write_to_CSV(food_web, filename):
    '''Writes foodweb to a CSV (spreadsheet) file.

    Parameters
    ----------
    foodweb : foodwebs.FoodWeb
        Object to save.
    filename: string
        Destination path.
    '''
    data = food_web.flow_matrix
    data = data.join(food_web.node_df[['IsAlive', 'Biomass', 'Export', 'Respiration', 'TrophicLevel']])
    data = data.append(food_web.node_df.Import)
    data = data.fillna(0.0)
    data.to_csv(filename, sep=';', encoding='utf-8')


def read_from_CSV(filename):
    '''Reads foodweb from a CSV (spreadsheet) file.

    Parameters
    ----------
    foodweb : foodwebs.FoodWeb
        Object to save.
    filename: string
        Destination path.
    '''
    data = pd.read_csv(filename, sep=';', encoding='utf-8').set_index('Names')

    if 'Import' not in data.index:
        raise Exception('Import row is missing.')

    imprt = data.loc[['Import']]
    data.drop('Import', inplace=True)

    node_columns = ['IsAlive', 'Biomass', 'Export', 'Respiration', 'TrophicLevel']
    for col in node_columns:
        if col not in data.columns:
            raise Exception(f'{col} column is missing.')
    node_df = data[node_columns].copy()

    imprt = imprt[[col for col in imprt.columns if col not in node_columns]]
    node_df['Import'] = imprt.values[0]

    if not all(node_df['IsAlive'].isin([1.0, 0.0])) or not all(node_df['IsAlive'].isin([True, False])):
        raise Exception('IsAlive column should have only True/False values.')

    node_df['IsAlive'] = node_df['IsAlive'].astype(bool)

    flow_matrix = data[[col for col in data.columns if col not in node_columns]]

    if (flow_matrix < 0).any().any():
        raise Exception('Flow matrix contains negative values.')

    if not np.array_equal(flow_matrix.columns, flow_matrix.index):
        raise Exception('Flow matrix (Internal flows sheet) should have exactly same rows as columns.')

    return fw.FoodWeb(title=filename.split('.csv')[0],
                      node_df=node_df.reset_index(),
                      flow_matrix=flow_matrix)


if __name__ == '__main__':
    f = read_from_SCOR('Alaska_Prince_William_Sound.scor')
    write_to_CSV(f, 'heh.csv')
    read_from_CSV('heh.csv')