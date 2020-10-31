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
from foodwebs.foodweb import FoodWeb
from foodwebs.foodweb_io import read_from_SCOR


from foodwebs.visualization import draw_trophic_flows_heatmap, show_trophic_flows_distribution

from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
# -

# ## Heatmap

draw_trophic_flows_heatmap(food_webs[0], log_scale=True)

# ## Fows distribution

show_trophic_flows_distribution(food_webs[0])

show_trophic_flows_distribution(food_webs[0], normalize=True)


