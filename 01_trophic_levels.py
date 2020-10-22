# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import altair as alt
import numpy as np
import pandas as pd
import seaborn as sns
from foodwebs.foodweb import FoodWeb
from foodwebs.foodweb_io import readFW_SCOR
import networkx as nx

import matplotlib.pyplot as pltś
import pylab

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8

# +
import glob
food_webs = [readFW_SCOR(net_path) for net_path in glob.glob('./data/*')]

for web in food_webs:
    print(f'{web.title[:30]} --> {web.n}, {web.n_living}')
    web.nodeDF['trophic'] = round(web.nodeDF.TrophicLevel)


# +
def get_trophic_level(net, name):
    ''' returns rounded trophic level for name '''
    return net.nodeDF[net.nodeDF.index == name].trophic.values[0]

def get_trophic_flows(net):
    ''' 
    returns sum of all flows weights between trophic levels 
    '''
    levels = sorted(net.nodeDF.trophic.unique())
    trophic_flows = {(x,y): 0 for x in levels for y in levels}
    
    for flow in net.getFlows(True):
        trophic_flows[(get_trophic_level(net, flow[0]), get_trophic_level(net, flow[1]))] += flow[2]

    # filter out zeros
    return [(x[0], x[1], float("{:.5f}".format(y))) for x, y in trophic_flows.items() if y != 0]


# -

get_trophic_flows(food_webs[0])

# ## Heatmap

# +
from collections import defaultdict

def get_trophic_flows(net):
    graph = net.getGraph()

    trophic_flows = defaultdict(float)
    for n, n_trophic in set(graph.nodes(data='TrophicLevel')):
        for m, m_trophic in set(graph.nodes(data='TrophicLevel')):
            weight = graph.get_edge_data(n, m, default=0)
            if weight != 0:
                trophic_flows[(round(n_trophic), round(m_trophic))] += weight['weight']
    return pd.DataFrame([(x, y, w) for (x, y), w in trophic_flows.items()], columns=['from', 'to', 'weights'])


# -

def draw_heatmap(net, log_scale=False):
    '''
    Draws heatmap of flows between trophic levels
    '''
    tf_pd = get_trophic_flows(net)
    if log_scale:
        tf_pd['weights'] = np.log(tf_pd.weights)
    tf_pd = tf_pd.pivot('from', 'to', 'weights')
    ax = sns.heatmap(tf_pd, annot=True, cmap="YlGnBu")


draw_heatmap(food_webs[0])  

draw_heatmap(food_webs[0], log_scale=True)


# ## Fows distribution

def show_trophic_flows_distribution(net, normalize=False):
    tf_pd = tf_pd = get_trophic_flows(net)
    tf_pd['to'] = tf_pd['to'].astype(str)

    return alt.Chart(tf_pd).mark_bar().encode(
        x=alt.X('sum(weights):Q', stack='zero' if not normalize else 'normalize'),
        y=alt.Y('from:N'),
        color=alt.Color('to'),
        order=alt.Order(
          # Sort the segments of the bars by this field
          'to',
          sort='ascending'
        )
    ).properties(width=1320, height=400)


show_trophic_flows_distribution(food_webs[0])

show_trophic_flows_distribution(food_webs[0], normalize=True)
