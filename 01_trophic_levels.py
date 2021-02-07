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
import foodwebs as fw

from foodwebs.foodweb import FoodWeb


from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
# -

# ## Heatmap

# +
import glob
food_webs = [fw.read_from_SCOR(net_path) for net_path in glob.glob('./data/*')]

for web in food_webs:
    print(f'{web.title[:30]} --> {web.n}, {web.n_living}')
# -

fw.draw_trophic_flows_heatmap(food_webs[0], log_scale=True)

# ## Fows distribution

fw.draw_trophic_flows_distribution(food_webs[0])

fw.draw_trophic_flows_distribution(food_webs[0], normalize=True)


