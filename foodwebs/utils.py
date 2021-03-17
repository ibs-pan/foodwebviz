'''Foodweb's utils methods.'''
import numpy as np
import pandas as pd


__all__ = [
    'NOT_ALIVE_MARK',
    'calculate_trophic_levels',
    'is_alive_mapping'
]


NOT_ALIVE_MARK = '\u2717'

def calculate_trophic_levels(food_web):
    
    
    '''Calculate the fractional trophic levels of nodes using their the recursive 
    relation. This implementation uses diet matrix to improve the numerical 
    behaviour of computation.
    In matrix form the trophic levels vector t is defined by
    t= 1 for primary producers and non-living nodes
    t= 1 + A t + b for other nodes
    where A_ij is a matrix of a fraction of node i diet a living node j contributes
    b=sum_k A_ik for k = non-living nodes

    Parameters
    ----------
    food web : foodwebs.FoodWeb
        Foodweb object.

    Returns
    -------
    trophic_levels : list
        List of trophic level values.
    '''
    data_size = len(food_web.flow_matrix)

    # the diagonal has the sum of all incoming system flows to the compartment i,
    # except flow from i to i
    A = food_web.get_diet_matrix().values.transpose() * -1
    A = A + np.full(data_size,1)
    tl=pd.DataFrame(food_web.flow_matrix.sum(axis=0), columns=['inflow'])
    #here we identify nodes at trophic level 1:
    tl['is_fixed_to_one'] = (tl.inflow <= 0.0) | (np.arange(data_size) >= food_web.n_living)
    tl['data_trophic_level'] = tl.is_fixed_to_one.astype(float)
    tl = tl.reset_index()

    # counting the nodes with TL fixed to 1
    if (sum(tl.is_fixed_to_one) != 0):
        not_one = tl[~tl.is_fixed_to_one].index.values
        one = tl[tl.is_fixed_to_one].index.values

        # update the equation due to the prescribed trophic level 1 - reduce the dimension of the matrix
        A_tmp = A[np.ix_(not_one, not_one)]

        B_tmp = tl[~tl.is_fixed_to_one]
        B_tmp = 1 #filling the constants vector with ones
        B_tmp -= np.sum(A[np.ix_(not_one, one)], axis=1) #this is the diet fraction fron non-living denoted as b in the function description

        A_inverse = np.linalg.pinv(A_tmp)
        A_inverse = np.multiply(A_inverse, B_tmp)
        tl.loc[~tl['is_fixed_to_one'], 'data_trophic_level'] = np.sum(A_inverse, axis=1)
    else:
        # fails with negative trophic levels = some problems
        np.linalg.pinv(A)
    return tl.data_trophic_level.values
    


def old_calculate_trophic_levels(food_web):
    '''Calculate trophic levels of nodes using their the recursive relation.

    Parameters
    ----------
    food web : foodwebs.FoodWeb
        Foodweb object.

    Returns
    -------
    trophic_levels : list
        List of trophic level values.
    '''
    data_size = len(food_web.flow_matrix)

    # sum of all incoming system flows to the compartment i
    inflow_pd = pd.DataFrame(food_web.flow_matrix.sum(axis=0), columns=['inflow'])

    # the diagonal has the sum of all incoming system flows to the compartment i,
    # except flow from i to i
    A = food_web.flow_matrix.values.transpose() * -1
    np.fill_diagonal(A, inflow_pd.values)

    inflow_pd['is_fixed_to_one'] = (inflow_pd.inflow <= 0.0) | (np.arange(data_size) >= food_web.n_living)
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


def is_alive_mapping(food_web):
    '''Creates dictionary which special X mark to names, which are not alive.

    Parameters
    ----------
    food web : foodwebs.FoodWeb
        Foodweb object.
    '''
    return {name: f'{NOT_ALIVE_MARK} {name}'
            for name in food_web.node_df[~food_web.node_df.IsAlive].index.values}
