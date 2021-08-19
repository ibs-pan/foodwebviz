# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 13:56:40 2021

Various utility functions, more universal in their possible applications.

@author: Mateusz
"""


def squeeze_map(x, min_x, max_x, map_fun, min_out, max_out):
    #we map the interval [min_x, max_x] into [min_out, max_out] so that the points are squeezed using the function map_fun
    # y_1 - y_2 ~ map_fun(c*x_1) - map_fun(c*x2) 
    #map_fun governs the mapping function, e.g. np.log10(), np.sqrt():
        #'sqrt': uses square root as the map basis
        #'log': uses log_10 as the map basis
    #a sufficient way to map an interval [a,b] (of flows) to a known interval [A, B] is:
        # f(x)= A + (B-A)g(x)/g(b) where g(a)=0
    # the implemented g(x) are log10, sqrt, but any strictly monotonously growing function mapping a to zero is fine
    return(min_out+ (max_out - min_out)*map_fun(x/min_x)/map_fun(max_x/min_x))

