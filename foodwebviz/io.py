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
    '''Reads a TXT file in the SCOR format and returns a FoodWeb object.

    Parameters
    ----------
    scor_path : string
        Path to the foodweb in SCOR format.


    Returns
    -------
    foodweb : foodwebs.Foodweb

    Description
    -------

    The SCOR format defines the graph through a list of edges (flows) and contains other food web data.
    Files in SCOR format look as follows (see examples/data/Richards_Bay_C_Summer.scor):
    --------------------
    title
    #of_all_nodes #of_living_nodes <-- size
    1st node name
    2nd node name
    ...
    1 biomass_of_the_1st_node
    2 biomass_of_the_2nd_node
    ...
    -1
    imports (same format as biomasses)
    -1
    exports (same format as biomasses)
    -1
    respirations (same format as biomasses)
    -1
    flows (in rows, e.g. '1 2 flow_from_1_to_2')
    -1
    --------------------

    Example:
    --------------------
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
    --------------------


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
        # reading vector input
        for i, col in enumerate(['Biomass', 'Import', 'Export', 'Respiration']):
            # each section should end with -1
            if lines[(i + 1) * n + i + n] != '-1':
                raise Exception(f'Invalid SCOR file format. {col} section could be wrong, \
                                  the separator -1 could be in a wrong place, names list \
                                  could have wrong length.')

            net[col] = [float(x.split(' ')[1])
                        for x in lines[(i + 1) * n + i: (i + 2) * n + i]]
        # reading the edge/flow list
        flow_matrix = pd.DataFrame(index=range(1, n+1), columns=range(1, n+1))
        for line in [x.split(' ') for x in lines[(i + 2) * n + i + 1:]]:
            if line[0].strip() == '-1':
                break
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

    Description
    --------
    SCOR format defines the graph through a list of edges (flows) and contains other food web data.
    Files in SCOR format look as follows:
    --------------------
    title
    #of_all_nodes #of_living_nodes <-- size
    1st node name
    2nd node name
    ...
    1 biomass_of_the_1st_node
    2 biomass_of_the_2nd_node
    ...
    -1
    imports (same format as biomasses)
    -1
    exports (same format as biomasses)
    -1
    respirations (same format as biomasses)
    -1
    flows (in rows, e.g. '1 2 flow_from_1_to_2')
    -1
    --------------------

    Example:
    --------------------
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
    --------------------
    '''
    def write_col(node_df, f, col):
        f.writelines([f'{i} {row}\n' for i, row in node_df[col].items()])
        f.write('-1\n')

    with open(scor_path, 'w') as f:
        f.write(f'{food_web.title}\n')
        f.write(f'{food_web.n} {food_web.n_living}\n')
        f.writelines([f'{x}\n' for x in food_web.node_df.index])

        node_df = food_web.node_df.reset_index().copy()
        node_df.index = node_df.index + 1
        n_map = node_df.reset_index().set_index('Names')[['index']].to_dict()['index']
        for col in ['Biomass', 'Import', 'Export', 'Respiration']:
            write_col(node_df, f, col)

        f.writelines(
            [f'{n_map[edge[0]]} {n_map[edge[1]]} {edge[2]["weight"]}\n' for edge in food_web.get_flows()])
        f.write('-1\n')
        f.write('\n')


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
    '''Read foodweb from an XLS (spreadsheet) file, see examples/data/Richards_Bay_C_Summer.xls.

    Parameters
    ----------
    foodweb : foodwebs.FoodWeb
        Object to save.
    filename: string
        Destination path.

    Description
    ----------
    The XLS file consists of three sheets:
        'Title':
            containing the name of the food web
        'Node properties':
            with a table describing nodes through the following columns:
            'Names', 'IsAlive', 'Biomass', 'Import', 'Export', 'Respiration', 'TrophicLevel'
        'Internal flows':
            with a table describing flows between the nodes in the system;
            the first row and the first column contain node names;
            table elements contain flow values from the node in the row to the node in the column.

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
                                   'Respiration': np.float64
                                   })
    flow_matrix = pd.read_excel(filename, sheet_name='Internal flows')
    if not np.array_equal(flow_matrix.columns.values[1:], flow_matrix.Names.values):
        raise Exception('Flow matrix (Internal flows sheet) should have exactly same rows as columns.')
    names = flow_matrix.Names
    flow_matrix.drop('Names', inplace=True, axis=1)
    flow_matrix.index = names
    flow_matrix.columns = names
    if (flow_matrix < 0).any().any():
        raise Exception('Flow matrix contains negative values.')
    return fw.FoodWeb(title=title.values[0][1], node_df=node_df, flow_matrix=flow_matrix)


def write_to_CSV(food_web, filename):
    '''Writes a food web to a CSV (spreadsheet) file, using semicolon as a separator.

    Parameters
    ----------
    food_web : foodwebs.FoodWeb
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
    '''Reads a food web from a CSV (spreadsheet) file.

    Parameters
    ----------

    filename: string
        Path to the CSV file. The expected format of a semicolon-separated file
        (see examples/data/Richards_Bay_C_Summer):
                Node 1;      Node 2;      ... Node N;      IsAlive; Biomass;   Export;	 Respiration
        Node 1; flow_1_to_1; flow_1_to_2; ... flow_1_to_N; 1;       biomass_1; export_1; Respiration_1
        Node 2; ...
        ...
        Node N; ...
        Import; import_1; ...

        The field IsAlive is 1 for living and 0 for non-living(detrital) nodes.
        Import, Export and Respiration encode the respective flows crossing the ecosystem boundary.

    Returns
    -------
    foodwebs.FoodWeb object

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
