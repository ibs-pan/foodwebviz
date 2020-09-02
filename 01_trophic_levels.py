# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md
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
    return net.nodeDF[net.nodeDF.Names == name].trophic.values[0]

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

def draw_heatmap(net, log_scale=False):
    '''
    Draws heatmap of flows between trophic levels
    '''
    tf_pd = pd.DataFrame(get_trophic_flows(net), columns=['from', 'to', 'weights'])
    if log_scale:
        tf_pd['weights'] = np.log(tf_pd.weights)
    tf_pd = tf_pd.pivot('from', 'to', 'weights')
    ax = sns.heatmap(tf_pd, annot=True, cmap="YlGnBu", fmt=".2f")


draw_heatmap(food_webs[0])

draw_heatmap(food_webs[0], log_scale=True)


# ## Fows distribution

def show_trophic_flows_distribution(net, normalize=False):
    tf_pd = pd.DataFrame(get_trophic_flows(net), columns=['from', 'to', 'weights'])
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

# ## Other

trophic_flows = get_trophic_flows(food_webs[0])

# +
import plotly.graph_objects as go

hist_pd = pd.DataFrame([(f'{x[0]} -> {x[1]}', x[2]) for x in trophic_flows], columns=['flow', 'weights'])

fig = go.Figure()
fig.add_trace(go.Histogram(histfunc="sum", y=hist_pd['weights'], x=hist_pd['flow'], name="sum"))
fig.show()

# +
import plotly.figure_factory as ff


# TODO funkcja z tego: miara podobieństwa, dane wejściowe różne
fig = ff.create_dendrogram(np.array([[x[2]] for x in trophic_flows]), orientation='left', labels=[f'{x[0]} -> {x[1]}' for x in trophic_flows])
fig.update_layout(width=800, height=500)
fig.show()
