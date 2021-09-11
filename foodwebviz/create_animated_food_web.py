# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:14:50 2021
The master functions for animation. 
@author: Mateusz
"""
import numpy as np
import matplotlib.pyplot as plt
import foodwebs as fw
import netImage as ni
from visualization import animate
from visuals import layer,add_vertices, init_particles, assign_colors

def adapt_dpi(node_nr):  # adapt the resolution to the number of nodes
    return(100+1.75*node_nr)


def adapt_particle_size(particles_nr, node_nr):  # adapt size of flow particles
    return(max(14000/particles_nr, 2))

#INPUT
def readNetImage(net_path, k, min_part_num, map_fun, max_part):  # read a netImage from a file
    net = fw.read_from_SCOR(net_path)
    return(ni.netImage(net, False, k_=k, min_part_num=min_part_num, map_fun=map_fun, max_part=max_part))
# def readNetImage(net_path,k, min_part_num): #read a netImage from a file
#     net = fw.read_from_SCOR(net_path)
#     return(ni.netImage(net,False,k_=k, min_part_num=min_part_num))


def adapt_node_and_font_size(n,maxwidth): #adapt the minimal node size to the number of nodes
    return((15/maxwidth,max(16,60/maxwidth)))
    #0.5 for 20 was ok
    #0.05 for 100?

#A food web example
scor_file_in = 'Alaska_Prince_William_Sound.scor'# 'atlss_cypress_wet.dat'
gif_file_out = 'Example.gif'

def animate_foodweb(scor_file_in, gif_file_out, fps=10, anim_len=1, trails=1, min_node_radius=0.5, max_node_radius=10, min_part_num=1, max_part_num=20, map_fun=np.sqrt, if_imports=True, if_exports=False, how_to_color='tl', cmap=plt.cm.get_cmap('viridis'), max_luminance=0.85):
    '''#foodweb_animation creates a GIF animation saved as gif_file_out based on the food web provided as a SCOR file scor_file_in. The canvas size in units relevant to further parameters is [0,100]x[0,100].
    #   trails:       the number of shades after each particle; shades are dots of diminishing opacity; it significantly impacts computation length
    #   min_node_radius: the radius of the smallest node on canvas [0,100]x[0,100]
    #   max_node_radius: the radius of the largest node on canvas [0,100]x[0,100]
    #   min_part_num: the number of particles representing the smallest flow
    #   max_part_num: the number of particles representing the largest flow
    #   map_fun: a function defined over (positive) real numbers and returning a positive real number;
    #                 used to squeeze the biomasses and flows that can span many orders of magnitude into the ranges
    #                 [min_node_radius, max_node_radius] and [min_part_num, max_part_num]
    #                 recommended functions: square root (numpy.sqrt) and logarithm (numpy.log10)
    #                   the map_fun is used as g(x) in mapping the interval [a,b] to a known interval [A, B]:
    #                   f(x)= A + (B-A)g(x)/g(b) where g(a)=0
    #   how_to_color: node coloring option, implemented
    #                 'tl' (according to the trophic level),
    #                 'discrete' (making neighbouring nodes of different colours);
    #   cmap:         colour map; must be continous for 'tl' coloring (e.g. seaborn.color_palette(as_cmap=True), matplotlib.pyplot.cm.get_cmap('viridis'))
    #                 and discrete for 'discrete' coloring (e.g. matplotlib.pyplot.cm.cubehelix)
    #   max_luminance: cut-off for the color range in cmap, in [0,1]: no cut-off is equivalent to max_luminance=1
    #                  usually, the highest values in a cmap range are very bright and hardly visible
'''
    def panimate(frame):
        plt.cla()
        layer(frame, particles, netIm, 1,t=interval+shadeStep*trails, how_to_color=how_to_color,particle_size=particle_size) #print the present positions of the particles
        #add vertices
        fig = plt.gcf()
        ax = fig.gca()
        add_vertices(ax, netIm.nodes,r_min=min_node_radius,r_max=max_node_radius, font_size=font_size, alpha_=0.95)
        T=0
        for i in range(trails): #shadow: alpha decreasing from 1 for i=0 to 0.2 at trails
            shadow_alpha=1-0.8*(i+1)/trails
            layer(frame, particles, netIm, shadow_alpha ,t=-shadeStep, how_to_color=how_to_color)
            T+=shadeStep
        layer(frame, particles, netIm, 0.8,t=T+shadeStep, how_to_color=how_to_color) #the final position
    
    interval = 0.3/fps  # time interval between frames
    shadeStep = interval/4  # what should be the distance of shades behind the actual particle position in terms of their time delay
    #how long should the animation be in s
    frameNumber = fps*anim_len  # number of frames
    #netIm=readNetImage('./examples/data/'+filename,80, min_part_num=min_part_num)# Iceland_227.dat')#BlackSea_433.dat')#
    #particles=init_particles(netIm,if_imports,if_exports,max_part_num)
    #particles=assign_colors(particles,netIm,how_to_color=how_to_color, max_luminance=max_luminance, cmap=cmap) #how_to_color='discrete' to paint each node differently for clarity
    netIm = readNetImage('../examples/data/'+scor_file_in, 80, min_part_num=min_part_num,
                         map_fun=map_fun, max_part=max_part_num)  # Iceland_227.dat')#BlackSea_433.dat')#
    particles = init_particles(netIm, if_imports, if_exports, max_part=max_part_num)
    # how_to_color='discrete' to paint each node differently for clarity
    particles = assign_colors(particles, netIm, how_to_color='trophic_level',
                              max_luminance=max_luminance, cmap=cmap)
    (max_node_radius,font_size)=adapt_node_and_font_size(len(netIm.nodes),np.max(netIm.nodes['width']))
    max_width = np.max(netIm.nodes['width'])
    (max_node_radius, font_size) = adapt_node_and_font_size(len(netIm.nodes), max_width)
    dpi = adapt_dpi(len(netIm.nodes))
    particle_size = adapt_particle_size(len(particles), len(netIm.nodes))
    animate(gif_file_out, panimate, 
        frames=frameNumber, interval=interval,figsize=(20, 20), dpi=dpi, fps=fps)#figsize=(12, 12)) #default dpi=100 #



#animate_foodweb(scor_file_in, 'Example.gif')#, anim_len=10, trails=7, min_part_num=8, max_part_num=150) for better quality




