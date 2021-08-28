# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 13:56:40 2021

Various utility functions, more universal in their possible applications.

@author: Mateusz
"""
from matplotlib import animation
import matplotlib.pyplot as plt


def squeeze_map(x, min_x, max_x, map_fun, min_out, max_out):
    #we map the interval [min_x, max_x] into [min_out, max_out] so that the points are squeezed using the function map_fun
    # y_1 - y_2 ~ map_fun(c*x_1) - map_fun(c*x2)
    #map_fun governs the mapping function, e.g. np.log10(), np.sqrt():
    #'sqrt': uses square root as the map basis
    #'log': uses log_10 as the map basis
    #a sufficient way to map an interval [a,b] (of flows) to a known interval [A, B] is:
    # f(x)= A + (B-A)g(x)/g(b) where g(a)=0
    # the implemented g(x) are log10, sqrt, but any strictly monotonously growing function mapping a to zero is fine
    return(min_out + (max_out - min_out)*map_fun(x/min_x)/map_fun(max_x/min_x))


def animate(filename, func, frames, interval, fig=None, figsize=(6.5, 6.5), fps=None, dpi=None):
    """ Creates an animated GIF of a matplotlib.

    Parameters
    ----------
    filename : string
        name of the file. E.g 'foo.GIF' or '\home\monty\parrots\fjords.gif'

    func : function
       function that will be called once per frame. Must have signature of
       def fun_name(frame_num)

    frames : int
       number of frames to animate. The current frame number will be passed
       into func at each call.

    interval : float
       Milliseconds to pause on each frame in the animation. E.g. 500 for half
       a second.

    figsize : (float, float)  optional
       size of the figure in inches. Defaults to 6.5" by 6.5"
    """

    def forward(frame):
        # I don't know if it is a bug or what, but FuncAnimation calls twice
        # with the first frame number. That breaks any animation that uses
        # the frame number in computations
        if forward.first:
            forward.first = False
            return
        func(frame)

    if fig is None:
        fig = plt.figure(figsize=figsize)

    #fig.patch.set_visible(False)

    forward.first = True
    anim = animation.FuncAnimation(fig, forward, frames=frames, interval=interval)
    anim.save(filename, writer='imagemagick', fps=fps, dpi=dpi)
