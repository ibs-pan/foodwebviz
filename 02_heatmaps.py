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
import math
import numpy as np
import pandas as pd
from foodwebs.foodweb import FoodWeb
from foodwebs.foodweb_io import read_from_SCOR
from foodwebs.normalization import flows_normalization
import networkx as nx

import plotly.graph_objects as go
import plotly.express as px
        
import matplotlib.pyplot as plt
import pylab

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8

# +
import glob
food_webs = [read_from_SCOR(net_path) for net_path in glob.glob('./data/*')]

for web in food_webs:
    print(f'{web.title[:30]} --> {web.n}, {web.n_living}')
# -

# ## Foodweb's heatmap

# ### no normalization

food_webs[2].show_heatmap(normalization='log', show_trophic_layer=True, add_external_flows=False)

food_webs[0].show_heatmap(normalization=None, show_trophic_layer=True, add_external_flows=True, switch_axes=False)

# ### biomass

food_webs[0].show_heatmap(normalization='biomass', show_trophic_layer=True, add_external_flows=False)

# ### log

food_webs[0].show_heatmap(normalization='log', show_trophic_layer=True, add_external_flows=False)

# ### diet

food_webs[0].show_heatmap(normalization='diet', show_trophic_layer=True, add_external_flows=True)

# # dendogram?

# +
import plotly.figure_factory as ff
import numpy as np
np.random.seed(1)

net = food_webs[5]

fig = ff.create_dendrogram(np.array([[x] for x in net.node_df.Biomass.values]), labels=net.node_df.index.values)
fig.update_layout(width=800, height=500)
fig.show()

# +
import plotly.figure_factory as ff
import numpy as np
np.random.seed(1)

net = food_webs[5]

fig = ff.create_dendrogram(np.array([[x] for x in net.node_df.Respiration.values]), labels=net.node_df.index.values)
fig.update_layout(width=800, height=500)
fig.show()

# +
net = food_webs[5]

fig = ff.create_dendrogram(net.node_df[['Biomass', 'IsAlive', 'Import', 'Export', 'Respiration', 'TrophicLevel']].values, orientation='left', labels=net.node_df.index.values)
fig.update_layout(width=1200, height=500)
fig.show()
