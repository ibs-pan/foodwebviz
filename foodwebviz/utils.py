'''Foodweb's utils methods.'''
import numpy as np
import pandas as pd


__all__ = [
    'NOT_ALIVE_MARK',
    'calculate_trophic_levels',
    'is_alive_mapping'
]


NOT_ALIVE_MARK = '\u2717'


def squeeze_map(x, min_x, max_x, map_fun, min_out, max_out):
    '''
    we map the interval [min_x, max_x] into [min_out, max_out] so that
    the points are squeezed using the function map_fun
    y_1 - y_2 ~ map_fun(c*x_1) - map_fun(c*x2)

    map_fun governs the mapping function, e.g. np.log10(), np.sqrt():
    'sqrt': uses square root as the map basis
    'log': uses log_10 as the map basis

    a sufficient way to map an interval [a,b] (of flows) to a known interval [A, B] is:
    f(x)= A + (B-A)g(x)/g(b) where g(a)=0

    the implemented g(x) are log10, sqrt, but any strictly
    monotonously growing function mapping a to zero is fine
    '''
    return min_out + (max_out - min_out) * map_fun(x / min_x) / map_fun(max_x / min_x)


def calculate_trophic_levels(food_web):
    '''Calculate the fractional trophic levels of nodes using their the recursive
    relation. This implementation uses diet matrix to improve the numerical
    behavior of computation.

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
    A = food_web.get_diet_matrix().transpose()

    tl = pd.DataFrame(food_web.flow_matrix.sum(axis=0), columns=['inflow'])
    # here we identify nodes at trophic level 1
    tl['is_fixed_to_one'] = (tl.inflow <= 0.0) | (np.arange(data_size) >= food_web.n_living)
    tl['data_trophic_level'] = tl.is_fixed_to_one.astype(float)

    # counting the nodes with TL fixed to 1
    if (sum(tl.is_fixed_to_one) != 0):
        # update the equation due to the prescribed trophic level 1 - reduce the dimension of the matrix
        A_tmp = A.loc[~tl['is_fixed_to_one'], ~tl['is_fixed_to_one']]
        A_tmp = A_tmp*-1 + pd.DataFrame(np.identity(len(A_tmp)), index=A_tmp.index, columns=A_tmp.columns)

        B = pd.DataFrame(tl[~tl.is_fixed_to_one].is_fixed_to_one.copy())
        # filling the constants vector with ones - the constant 1 contribution
        B['b'] = 1
        # this is the diet fraction from non-living denoted as b in the function description
        B['b'] = B['b'] + A.loc[~tl['is_fixed_to_one'], tl['is_fixed_to_one']].sum(axis=1)

        A_inverse = np.linalg.pinv(A_tmp)
        tl.loc[~tl['is_fixed_to_one'], 'data_trophic_level'] = np.dot(A_inverse, B['b'])
    else:
        # fails with negative trophic levels = some problems
        np.linalg.pinv(A)
    return tl.data_trophic_level.values


def is_alive_mapping(food_web):
    '''Creates dictionary which special X mark to names, which are not alive.

    Parameters
    ----------
    food web : foodwebs.FoodWeb
        Foodweb object.
    '''
    return {name: f'{NOT_ALIVE_MARK} {name}'
            for name in food_web.node_df[~food_web.node_df.IsAlive].index.values}
