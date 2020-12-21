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
from foodwebs.foodweb import FoodWeb, NOT_ALIVE_MARK
from foodwebs.foodweb_io import read_from_SCOR
from foodwebs.normalization import flows_normalization
import networkx as nx
from pyvis.network import Network

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

food_webs[1].show_network_for_nodes(['Red mud crab'])
