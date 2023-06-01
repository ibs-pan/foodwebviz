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

# global velocity along the lines so that it loops back on itself
VELOCITY = 0.1  # 1/(5*ANIMATION_LEGHT)


BOX_PARAMS = {'facecolor': 'white',
              'alpha': 0.7,
              'edgecolor': 'white',
              'pad': 0.1}


def particles_in_one_flow(flows, x1, x2, y1, y2, start_node, max_part, map_fun):
    '''
    distribute particles

    return particles moving from (x1,y1) to (x2, y2) and their start_node saved to define color later
    spaced randomly (uniform dist) along a line defined by start and finish
    with s in [0,1] tracing their progress along the line
    '''
    def stretch(a, b):
        return np.full(int(b), a)

    lx = x2 - x1
    ly = y2 - y1

    # we need to normalize to path length
    flow_density = int(flows * np.sqrt(lx**2 + ly**2) / 20)
    s = uniform(0, 1, flow_density)

    # we spread the particles randomly in direction perpendicular to the line
    # making larger flows broader
    width = squeeze_map(flows, 1, max_part, map_fun, 0.05, 3)

    x1_new = stretch(x1, flow_density)
    y1_new = stretch(y1, flow_density)

    # spread them randomly
    x1_new += uniform(-width / 2, width / 2, flow_density) if ly != 0.0 else 0.0
    y1_new += uniform(-width / 2, width / 2, flow_density) if lx != 0.0 else 0.0

    return pd.DataFrame({'s': s,
                         'x': x1_new + s * lx,
                         'y': y1_new + s * ly,
                         'x1': x1_new,
                         'y1': y1_new,
                         'lx': lx,
                         'ly': ly,
                         'start': start_node})


def init_particles(network_image, include_imports, include_exports, max_part, map_fun):
    '''
    given the network image with node positions
    and the number of particles flowing between them, initialize particles
    '''
    xs = network_image.nodes.x
    ys = network_image.nodes.y

    # number of particles along a system flow
    partNumber_sys_flows, partNumber_imports, partNumber_exports = network_image.particle_numbers

    particles = pd.DataFrame()
    for i in xs.index:
        for j in xs.index:
            # first the system flows
            if partNumber_sys_flows.loc[i, j] != 0.0:  # we do nothing for zero flows
                particles = particles.append(
                    particles_in_one_flow(partNumber_sys_flows.loc[i, j], xs[i], xs[j], ys[i], ys[j],
                                          start_node=i, max_part=max_part, map_fun=map_fun))

        if include_imports:
            particles = particles.append(
                particles_in_one_flow(partNumber_imports.loc[i], xs[i], xs[i], 0.0, ys[i],
                                      start_node=i, max_part=max_part, map_fun=map_fun))
        if include_exports:
            particles = particles.append(
                particles_in_one_flow(partNumber_exports.loc[i], xs[i],
                                      0.0 if xs[i] < 50 else 100.0, ys[i], ys[i],
                                      start_node=i, max_part=max_part, map_fun=map_fun))
    return particles.reset_index()


def _get_color_for_trophic_level(df, y, max_luminance, cmap):
    # uses only the trophic level to define color on a continuous scale
    def set_color(z, minVal, maxVal, max_luminance=0.85):
        return np.interp(z, (minVal, maxVal), (0, max_luminance))
    return [cmap(x) for x in set_color(df[y], df[y].min(), df[y].max(), max_luminance)]


def assign_colors(particles, netIm, max_luminance, cmap):
    '''
    specify colors using coordinates in columns x and y of the dataframe df
    '''
    netIm.nodes['color'] = _get_color_for_trophic_level(netIm.nodes, 'y', max_luminance, cmap=cmap)
    particles['color'] = particles.apply(lambda x: netIm.nodes.loc[x['start'], 'color'], axis='columns')
    return particles


# # RULES TO UPDATE FRAMES
# def getVelocity(row, lx, ly):  # get axial velocity along lx given the direction (lx,ly)
#     return(VELOCITY*row[lx]/np.sqrt(row[lx]**2+row[ly]**2))


def move_particles(particles, alpha, t, max_width):
    def accelerate_imports(row):
        '''
        imports have shorter way to go, we make them go faster to be noticed
        unless they are at higher trophic levels
        '''
        return 4 * VELOCITY if row['lx'] == 0 and np.abs(row['ly']) < 20 else 0.0

    def fading_formula(x, max_width):
        '''
        how the position along the edge s in [0,1] is translated into alpha (transparency) value
        we adapt the fading to max_width as a proxy for the complexity of the network
        '''
        # we shift the transparency by the minimal value that is attained in the middle
        min_alpha = 1 / max_width
        exponent_correction = 2 * int(max_width / 8)

        # 1 at ends, 0.5 in the middle, parabolic dependence
        return max([min([1, (np.abs(x - 0.5)**(2 + exponent_correction)) * 2**(2 + exponent_correction)]),
                    min_alpha])

    # updating s and cycling within [0,1] Old case of constant progress over line
    particles['s'] = (particles['s'] + VELOCITY * t +
                      particles.apply(accelerate_imports, axis='columns') * t) % 1

    # which we save translated to 'x' and 'y'
    particles['x'] = particles['x1'] + particles['lx'] * particles['s']
    particles['y'] = particles['y1'] + particles['ly'] * particles['s']

    # we make particles fade a bit when far from both the source and the target
    particles['alpha'] = alpha * particles['s'].apply(fading_formula, max_width=max_width)


# adds a vertex in position x,y with biomass b to axis ax, given the largest biomass maxBio
def _add_vertex(row, ax, min_bio, max_bio, r_min, r_max, font_size,
                alpha, map_fun=np.log10, list_of_abbrev=[]):
    radius = squeeze_map(row['Biomass'], min_bio, max_bio, map_fun, r_min, r_max)

    name = row['Names'].replace('PrimProd', '').strip()
    if len(name) > 16:
        old_name = name

        # abbreviate long names
        name = ' '.join(map(lambda x: x if len(x) <= 3 else f'{x[:3]}.', name.split(' ')))
        list_of_abbrev.append(f'{name} = {old_name}\n')

    vert_shift = 0.08 if (row['x_rank'] % 2) == 1 or (row['TrophicLevel'] == 1.0 and font_size > 20) else -0.1

    txt = plt.text(max(row['x'] - len(name) * 0.03 * font_size, 1),
                   min(row['y'] + np.sign(vert_shift) * radius + vert_shift * font_size, 98),
                   name,
                   fontsize=font_size)
    txt.set_bbox(BOX_PARAMS)
    ax.add_patch(plt.Circle((row['x'], row['y']), radius, color=row['color'], alpha=alpha))


def add_vertices(ax, yDf, r_min, r_max, font_size, alpha, map_fun=np.log10):
    '''
    adds circular vertices to the axis ax
    '''
    yDf.sort_values(by='x')
    yDf.sort_values(by='y')

    list_of_abbrev = []
    yDf.apply(_add_vertex,
              axis='columns',
              ax=ax,
              r_min=r_min,
              r_max=r_max,
              min_bio=np.min(yDf['Biomass']),
              max_bio=np.max(yDf['Biomass']),
              font_size=font_size,
              alpha=alpha,
              map_fun=map_fun,
              list_of_abbrev=list_of_abbrev)
    list_of_abbrev.sort()

    abbrev_leg = plt.text(100, 17, f"Abbreviations used:\n{''.join(list_of_abbrev)}",
                          fontsize=font_size,
                          horizontalalignment='right',
                          verticalalignment='bottom')
    abbrev_leg.set_bbox(BOX_PARAMS)


def create_layer(frame, particles,  netIm, alpha, t=INTERVAL_BETWEEN_FRAMES, max_width=8, particle_size=2):
    def add_transparency_to_color(particle_row):
        '''
        set transparency within RGBA colours
        '''
        new_color = list(particle_row['color'])
        # continuous cmaps have longer tuple with alpha as the fourth item:
        new_color[3] = particle_row['alpha']
        return tuple(new_color)

    move_particles(particles, alpha, t, max_width)

    plt.scatter(particles['x'], particles['y'],
                s=particle_size,
                # make particles fade except when around their target or start nodes
                c=particles.apply(add_transparency_to_color, axis='columns'),
                edgecolors={'none'})

    # Create a new colormap from the colors cut to 0.8 (to avoid too light colors)

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.gca().axis('off')
