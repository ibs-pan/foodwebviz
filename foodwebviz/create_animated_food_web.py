# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:14:50 2021
The master functions for animation.
@author: Mateusz
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from foodwebviz.animation.network_image import NetworkImage
from foodwebviz.animation import animation_utils


__all__ = [
    'animate_foodweb',
]


def _run_animation(filename, func, frames, interval, fig=None, figsize=(6.5, 6.5), fps=None, dpi=None):
    r""" Creates an animated GIF of a matplotlib.

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

    forward.first = True
    anim = animation.FuncAnimation(fig, forward, frames=frames, interval=interval)
    anim.save(filename, writer='imagemagick', fps=fps, dpi=dpi)


def animate_foodweb(foodweb, gif_file_out, fps=10, anim_len=1, trails=1,
                    min_node_radius=0.5, min_part_num=1,
                    max_part_num=20, map_fun=np.sqrt, include_imports=True, include_exports=False,
                    cmap=plt.cm.get_cmap('viridis'), max_luminance=0.85,
                    particle_size=8):
    '''foodweb_animation creates a GIF animation saved as gif_file_out based on the food web
        provided as a SCOR file scor_file_in. The canvas size in units relevant
        to further parameters is [0,100]x[0,100].

        Parameters
        ----------
        trails : int
            the number of shades after each particle; shades are dots of diminishing opacity;
            it significantly impacts computation length
        min_node_radius : float
            the radius of the smallest node on canvas [0,100]x[0,100]
        min_part_num : int
            the number of particles representing the smallest flow
        max_part_num : int
            the number of particles representing the largest flow
        map_fun : function
            a function defined over (positive) real numbers and returning a positive real number;
            used to squeeze the biomasses and flows that can span many orders of magnitude into the ranges
            [min_node_radius, max_node_radius] and [min_part_num, max_part_num]
            recommended functions: square root (numpy.sqrt) and logarithm (numpy.log10)
            the map_fun is used as g(x) in mapping the interval [a,b] to a known interval [A, B]:
            f(x)= A + (B-A)g(x)/g(b) where g(a)=0
        cmap :
            a continuous colour map
            (e.g. seaborn.color_palette("cubehelix", as_cmap=True), matplotlib.pyplot.cm.get_cmap('viridis'))
        max_luminance : float
            cut-off for the color range in cmap, in [0,1]: no cut-off is equivalent to max_luminance=1
            usually, the highest values in a cmap range are very bright and hardly visible
        particle_size: float
            size of the flow particles
    '''
    def animate_frame(frame):
        plt.cla()
        # print the present positions of the particles
        animation_utils.create_layer(frame, particles, network_image, 1, t=interval + shade_step * trails,
                                     particle_size=particle_size)

        fig = plt.gcf()
        ax = fig.gca()
        animation_utils.add_vertices(ax, network_image.nodes, r_min=min_node_radius,
                                     r_max=max_node_radius, font_size=font_size, alpha=0.95)
        T = 0
        for i in range(trails):  # shadow: alpha decreasing from 1 to the lowest specified for flows
            shadow_alpha = 1-0.5*(i+1)/trails
            animation_utils.create_layer(frame, particles, network_image, shadow_alpha,
                                         t=-shade_step, particle_size=shadow_alpha*particle_size)
            T += shade_step

    # time interval between frames
    interval = 0.3 / fps

    # what should be the distance of shades behind the actual particle position in terms of their time delay
    shade_step = 0.7 * interval

    # create a static graph representation of the food web
    # and map flows and biomass to particle numbers and node sizes
    network_image = NetworkImage(foodweb, False, k_=80,
                                 min_part_num=min_part_num,
                                 map_fun=map_fun,
                                 max_part=max_part_num)

    particles = animation_utils.init_particles(network_image, include_imports, include_exports,
                                               max_part=max_part_num, map_fun=map_fun)
    particles = animation_utils.assign_colors(particles, network_image,
                                              max_luminance=max_luminance, cmap=cmap)

    # adapt the minimal node size to the number of nodes
    max_width = network_image.nodes.width.max()
    max_node_radius = 15 / max_width
    font_size = max(10, 60 / max_width)

    _run_animation(gif_file_out,
                   func=animate_frame,
                   frames=fps * anim_len,  # number of frames,
                   interval=interval,
                   figsize=(20, 20),
                   dpi=100 + 1.75 * len(network_image.nodes),  # adapt the resolution to the number of nodes
                   fps=fps)
