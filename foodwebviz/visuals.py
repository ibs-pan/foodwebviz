# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 13:15:24 2021

Animates streams of points/particles given their starting and ending locations
and number of particles in  each flow.

@author: Mateusz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import uniform

from foodwebviz.utils import squeeze_map

# here some global design choices

FPS = 60

# time between frames in ms
INTERVAL_BETWEEN_FRAMES = 1000 / FPS

# how long should the animation be in s
ANIMATION_LEGHT = 1

# number of frames in ms
NUMBER_OF_FRAMES = FPS*ANIMATION_LEGHT

# global velocity along the lines so that it loops back on itself
VELOCITY = 0.1  # 1/(5*ANIMATION_LEGHT)


def stretch(x, l):
    return(np.full(int(l), x))


# distribute particles
def particles_in_one_flow(nOne_, x1, x2, y1, y2, start_node, max_part, map_fun):
    # return particles moving from (x1,y1) to (x2, y2) and their start_node saved to define color later
    # spaced randomly (uniform dist) along a line defined by start and finish
    # with s in [0,1] tracing their progress along the line
    lx = (x2-x1)
    ly = (y2-y1)
    # we want the nOne to represent flow density, so we need to normalize to path length
    nOne = int(nOne_*np.sqrt(lx**2+ly**2)/20)
    s = uniform(0, 1, nOne)

    # we spread the particles randomly in direction perpendicular to the line
    # making larger flows broader
    # lxd = np.interp(nOne_, [0, max_part], [0, 1.5])  # max(lx/20,3)
    width = squeeze_map(nOne_, 1, max_part, map_fun, 0.05, 3)

    distortX = uniform(-width/2, width/2, nOne) # spread them randomly
    distortY = uniform(-width/2, width/2, nOne)
    x1_ = stretch(x1, nOne)
    y1_ = stretch(y1, nOne)
    if ly != 0.0:
        x1_ += distortX
    if lx != 0.0:
        y1_ += distortY
    # x1_=x1
    # y1_=y1
    d = {'s': s, 'x': x1_+s*lx, 'y': y1_+s*ly, 'x1': x1_, 'y1': y1_, 'lx': lx, 'ly': ly, 'start': start_node}
    return pd.DataFrame(d)


def init_particles(netIm, if_imports, if_exports, max_part, map_fun):
    # given the network image with node positions
    # and the number of particles flowing between them, initialize particles
    xs = netIm.nodes.x
    ys = netIm.nodes.y
    # number of particles along a system flow
    [partNumber_sys_flows, partNumber_imports, partNumber_exports] = netIm.particle_numbers

    particles = pd.DataFrame()
    for i in xs.index:
        for j in xs.index:
            #first the system flows
            if partNumber_sys_flows.loc[i, j] != 0.0:  # we do nothing for zero flows
                particles = pd.concat([particles, particles_in_one_flow(
                    partNumber_sys_flows.loc[i, j], xs[i], xs[j], ys[i], ys[j], start_node=i, max_part=max_part, map_fun=map_fun)])
        if if_imports:
            particles = pd.concat([particles, particles_in_one_flow(
                partNumber_imports.loc[i], xs[i], xs[i], 0.0, ys[i], start_node=i, max_part=max_part, map_fun=map_fun)])
        if if_exports:
            if xs[i] < 50:
                particles = pd.concat([particles, particles_in_one_flow(
                    partNumber_exports.loc[i], xs[i], 0.0, ys[i], ys[i], start_node=i, max_part=max_part, map_fun=map_fun)])
            else:
                particles = pd.concat([particles, particles_in_one_flow(
                    partNumber_exports.loc[i], xs[i], 100.0, ys[i], ys[i], start_node=i, max_part=max_part, map_fun=map_fun)])
    particles = particles.reset_index()
    return(particles)


# SPECIFY MATCHING COLORS OF ELEMENTS
def getSqrt(row, x, y):
    # computes the color based on colormap cm and two dataframe columns
    return(np.sqrt(row[x]**2+row[y]**2))


def setColor(z_, minVal, maxVal, max_luminance=0.85):
    z = np.interp(z_, (minVal, maxVal), (0, max_luminance))
    return(z)


# specify colors using coordinates in columns x and y of the dataframe df
def getColors_continuous(df, x, y, max_luminance, cmap):
    z = df.apply(getSqrt, axis='columns', x=x, y=y)  # earlier take on continuous palette
    return([cmap(x) for x in setColor(z, z.min(), z.max(), max_luminance)])  # version with continuous palette


def get_color_for_tl(df, y, max_luminance, cmap):
    # uses only the trophic level to define color on a continuous scale
    return([cmap(x) for x in setColor(df[y], df[y].min(), df[y].max(), max_luminance)])


# specify colors using coordinates in columns x and y of the dataframe df
def assign_colors(particles, netIm, max_luminance, cmap):
    netIm.nodes['color'] = get_color_for_tl(netIm.nodes, 'y', max_luminance, cmap=cmap)
    particles['color'] = particles.apply(lambda x: netIm.nodes.loc[x['start'], 'color'], axis='columns')
    return(particles)


# RULES TO UPDATE FRAMES
def getVelocity(row, lx, ly):  # get axial velocity along lx given the direction (lx,ly)
    return(VELOCITY*row[lx]/np.sqrt(row[lx]**2+row[ly]**2))


# how the position along the edge s in [0,1] is translated into alpha (transparency) value
def fading_formula(x, max_width):
    #we adapt the fading to max_width as a proxy for the complexity of the network
    min_alpha = 1/max_width  # we shift the transparency by the minimal value that is attained in the middle
    exponent_correction=2*int(max_width/8)
    return(max([min([1, (np.abs(x-0.5)**(2+exponent_correction))*2**(2+exponent_correction)]), min_alpha]))  # 1 at ends, 0.5 in the middle, parabolic dependence


# we make particles fade a bit when far from both the source and the target
def update_transparency(particles, overall_alpha, max_width):
    particles['alpha'] = overall_alpha*particles['s'].apply(fading_formula, max_width=max_width)
    return(particles)


def accelerate_imports(row):  # imports have shorter way to go, we make them go faster to be noticed
    #unless they are at higher trophic levels
    if row['lx'] == 0 and np.abs(row['ly']) < 20:
        return(4*VELOCITY)
    else:
        return(0.0)


def move(particles, alpha_, t, max_width):
    # we move a particle along its line by updating 's'
    # vx=particles.apply(getVelocity,axis='columns',lx='lx',ly='ly')
    # vy=particles.apply(getVelocity,axis='columns',lx='ly',ly='lx')

    # updating s and cycling within [0,1] Old case of constant progress over line
    particles.loc[:, 's'] = (particles.loc[:, 's']+VELOCITY*t +
                             particles.apply(accelerate_imports, axis='columns')*t) % 1
    # which we save translated to 'x' and 'y'

    # np.mod(particles.loc[:,'x']-particles.loc[:,'x1'] + vx*t , particles.loc[:,'lx'])
    particles.loc[:, 'x'] = particles.loc[:, 'x1'] + particles.loc[:, 'lx']*particles.loc[:, 's']
    # np.mod(particles.loc[:,'y']-particles.loc[:,'y1'] + vy*t , particles.loc[:,'ly'])
    particles.loc[:, 'y'] = particles.loc[:, 'y1']+particles.loc[:, 'ly']*particles.loc[:, 's']
    #particles.loc[:,'s']=np.sqrt((particles.loc[:,'x']-particles.loc[:,'x1'])**2+(particles.loc[:,'y']-particles.loc[:,'y1'])**2/(particles.loc[:,'lx']**2+particles.loc[:,'ly']**2))
    particles = update_transparency(particles, alpha_, max_width)


def get_color_with_transparency(particle_row):  # set transparency within RGBA colours
    new_color = list(particle_row['color'])
    # continuous cmaps have longer tuple with alpha as the fourth item:
    new_color[3] = particle_row['alpha']
    return(tuple(new_color))

# Adding vertices
def abbrev_word(s):
    if len(s) > 3:
        return(s[:3]+'.')
    else:
        return(s)


def abbreviate(s):  # abbreviate long names
    words = s.split(' ')
    new_words = map(abbrev_word, words)

    return(' '.join(new_words))


# adds a vertex in position x,y with biomass b to axis ax, given the largest biomass maxBio
def addVertex(row, ax, min_bio, max_bio, r_min, r_max, font_size, alpha_, map_fun=np.log10, list_of_abbrev=[]):
    radius = squeeze_map(row['bio'], min_bio, max_bio, map_fun, r_min, r_max)
    #print('radius: '+str(radius))
    circle = plt.Circle((row['x'], row['y']), radius, color=row['color'], alpha=alpha_)
    name = row['name'].replace('PrimProd', '').strip()
    if len(name) > 16:
        old_name = name
        name = abbreviate(name)
        list_of_abbrev.append(name+' = '+old_name+'\n')
    if (row['x_rank'] % 2) == 1 or (row['TL'] == 1.0 and font_size > 20):
        vert_shift = 0.08
    else:
        vert_shift = -0.1
    txt = plt.text(max(row['x']-len(name)*0.03*font_size, 1), min(row['y'] +
                                                                  np.sign(vert_shift)*radius+vert_shift*font_size, 98), name, fontsize=font_size)
    txt.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white', pad=0.1))
    ax.add_patch(circle)


def add_vertices(ax, yDf, r_min, r_max, font_size, alpha_, map_fun=np.log10):  # adds circular vertices to the axis ax

    yDf.sort_values(by='x')
    yDf.sort_values(by='y')
    list_of_abbrev = []
    yDf.apply(addVertex, axis='columns', ax=ax, r_min=r_min, r_max=r_max, min_bio=np.min(yDf['bio']), max_bio=np.max(
        yDf['bio']), font_size=font_size, alpha_=alpha_, map_fun=map_fun, list_of_abbrev=list_of_abbrev)
    list_of_abbrev.sort()
    abbrev_leg = plt.text(100, 17, 'Abbreviations used:\n'+''.join(list_of_abbrev),
                          fontsize=font_size, horizontalalignment='right',  verticalalignment='bottom')
    abbrev_leg.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white', pad=0.1))


def add_trophic_level_legend(ax, pos_df, font_size):  # adds a legend of trophic levels

    levels = np.interp([list(range(1, int(pos_df['TL'].max()+1)))], (pos_df['TL'].min(),
                                                                     pos_df['TL'].max()), (5, 95))  # this is how we map trophic levels to positions
    i = 1
    for level in levels:
        print(level)
        txt = plt.text(2, level, 'Trophic level '+str(i), fontsize=2*font_size,
                       horizontalalignment='left', verticalalignment='center')
        txt.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor='white', pad=0.1))
        i += 1


def layer(frame, particles,  netIm, alpha_, t=INTERVAL_BETWEEN_FRAMES, max_width=8, particle_size=2):
    # print('Frame with alpha '+str(alpha_)+' step '+str(t))
    move(particles, alpha_, t, max_width)
    # make particles fade except when around their target or start nodes
    rgba_colors = particles.apply(get_color_with_transparency, axis='columns')
    plt.scatter(particles.loc[:, 'x'], particles.loc[:, 'y'],
                s=particle_size, c=rgba_colors, edgecolors={'none'})
    # Create a new colormap from the colors cut to 0.8 (to avoid too light colors)

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.gca().axis('off')
    # plt.gca().get_xaxis().set_ticks([])
    # plt.gca().get_yaxis().set_ticks([1,2,3,4,5])
