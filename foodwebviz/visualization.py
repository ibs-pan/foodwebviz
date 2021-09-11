'''Foodweb's visualization methods.'''
import decimal
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

from pyvis.network import Network
from collections import defaultdict
import foodwebviz as fw


__all__ = [
    'draw_heatmap',
    'draw_trophic_flows_heatmap',
    'draw_trophic_flows_distribution',
    'draw_network_for_nodes'
]


decimal.getcontext().rounding = decimal.ROUND_HALF_UP

TROPHIC_LAYER_COLORS = [
    [0, 'rgb(255, 255, 255)'],
    [0.2, 'rgb(214, 233, 255)'],
    [0.4, 'rgb(197, 218, 251)'],
    [0.6, 'rgb(182, 201, 247)'],
    [0.8, 'rgb( 168, 183, 240 )'],
    [1.0, 'rgb(  167, 167, 221 )']
]

HEATMAP_COLORS = [
    [0.0, 'rgb(222, 232, 84)'],
    [0.2, 'rgb( 117, 188, 36)'],
    [0.4, 'rgb( 27, 167, 50 )'],
    [0.6, 'rgb( 28, 125, 57 )'],
    [0.8, 'rgb(59, 28, 95)'],
    [1.0, 'rgb(27, 15, 36 )']
]


def _get_title(food_web, limit=150):
    return food_web.title if len(food_web.title) <= limit else food_web.title[:limit] + '...'


def _get_trophic_layer(graph, from_nodes, to_nodes):
    '''Creates Trace for Heatmap to show thropic levels of X axis nodes.

    Parameters
    ----------
    graph : networkx.SubGraph
        Graph View representing foodweb.
    from_nodes: list
        List of 'from' nodes.
    to_nodes: list
        List of 'to' nodes.

    Returns
    -------
    trophic_layer : plotly.graph_objects.Heatmap
    '''
    trophic_flows = []
    for n in set(from_nodes):
        trophic_flows.extend([(n, m, graph.nodes(data='TrophicLevel', default=0)[n]) for m in set(to_nodes)])

    fr, to, z = list(zip(*trophic_flows))
    return go.Heatmap(
        z=z,
        x=to,
        y=fr,
        showlegend=True,
        showscale=False,
        xgap=0.2,
        ygap=0.2,
        zmin=min(z),
        zmax=max(z) + 3,
        colorscale=TROPHIC_LAYER_COLORS,
        name='Trophic Layer',
        hoverinfo='skip'
    )


def _get_trophic_flows(food_web, normalization='linear'):
    '''For each pair of trophic levels assigns sum of all nodes' weights in that pair.

    Parameters
    ----------
    food_web : foodwebs.FoodWeb
        Foodweb object.
    normalization : string, optional (default=linear)
        Defines method of graph edges normalization.
        Available options are: 'diet', 'log', 'donor_control',
        'predator_control', 'mixed_control', 'linear' and 'tst'.

    Returns
    -------
    trophic_flows : pd.DataFrame
        Columns: ["from", "to", "wegiths"], where "from" and "to" are trophic levels.
    '''
    graph = food_web.get_graph(False, mark_alive_nodes=False, normalization=normalization)

    trophic_flows = defaultdict(float)
    trophic_levels = {node: level for node, level in graph.nodes(data='TrophicLevel')}
    for edge in graph.edges(data=True):
        trophic_from = decimal.Decimal(trophic_levels[edge[0]]).to_integral_value()
        trophic_to = decimal.Decimal(trophic_levels[edge[1]]).to_integral_value()
        trophic_flows[(trophic_from, trophic_to)] += edge[2]['weight']

    return pd.DataFrame([(x, y, w) for (x, y), w in trophic_flows.items()], columns=['from', 'to', 'weights'])


def _get_array_order(graph, nodes, reverse=False):
    def sort_key(x): return (x[1].get('TrophicLevel', 0), x[1].get('IsAlive', 0))
    return [x[0] for x in sorted(graph.nodes(data=True), key=sort_key, reverse=reverse) if x[0] in nodes]


def draw_heatmap(food_web, boundary=False, normalization='log',
                 show_trophic_layer=True, switch_axes=False,
                 width=1200, height=800):
    '''Visualize foodweb as a heatmap. On the interesction
    of X axis ("from" node) and Y axis ("to" node) flow weight
    is indicated.

    Parameters
    ----------
    food_web : foodwebs.FoodWeb
        Foodweb object.
    boundary : bool, optional (default=False)
        If True, boundary flows will be added to the graph.
        Boundary flows are: Import, Export, and Repiration.
    normalization : string, optional (default=log)
        Defines method of graph edges normalization.
        Available options are: 'diet', 'log', 'donor_control',
        'predator_control', 'mixed_control', 'linear' and 'tst'.
    show_trophic_layer : bool, optional (default=False)
        If True, include additional heatmap layer presenting trophic levels relevant to X axis.
    switch_axes : bool, optional (default=False)
        If True, X axis will represent "to" nodes and Y - "from".
    width : int, optional (default=1200)
        Width of the heatmap plot, large foodwebs might not fit in default width.
    height : int, optional (default=800)
        Height of the heatmap plot, large foodwebs might not fit in default height.

    Returns
    -------
    heatmap : plotly.graph_objects.Figure
    '''

    graph = food_web.get_graph(boundary, mark_alive_nodes=True, normalization=normalization)
    if switch_axes:
        to_nodes, from_nodes, z = list(zip(*graph.edges(data=True)))
        hovertemplate = '%{x} --> %{y}: %{z:.3f}<extra></extra>'
    else:
        from_nodes, to_nodes, z = list(zip(*graph.edges(data=True)))
        hovertemplate = '%{y} --> %{x}: %{z:.3f}<extra></extra>'

    z = [w['weight'] for w in z]

    fig = go.Figure()
    if show_trophic_layer:
        fig.add_trace(_get_trophic_layer(graph, from_nodes, to_nodes))

    heatmap = go.Heatmap(
        z=z,
        x=to_nodes,
        y=from_nodes,
        showlegend=False,
        showscale=True,
        xgap=0.2,
        ygap=0.2,
        zmin=min(z),
        zmax=max(z),
        colorscale=HEATMAP_COLORS,
        hoverongaps=False,
        hovertemplate='%{y} --> %{x}: %{z:.3f}<extra></extra>'
    )

    # fix color bar for log normalization
    if normalization == 'log':
        z_orginal = [x[2]['weight'] for x in food_web.get_graph(
            boundary, mark_alive_nodes=True, normalization=None).edges(data=True)]
        tickvals = range(int(np.log10(min(z_orginal))) + 1, int(np.log10(max(z_orginal))) + 1)
        ticktext = [10**x for x in range(len(tickvals))]
 
        heatmap.colorbar = dict(
            tick0=0,
            tickmode='array',
            tickvals=list(tickvals),
            ticktext=ticktext
        )
        heatmap.customdata = z_orginal
        if switch_axes:
            hovertemplate = '%{x} --> %{y}: %{customdata:.3f}<extra></extra>'
        else:
            hovertemplate = '%{y} --> %{x}: %{customdata:.3f}<extra></extra>'
        heatmap.hovertemplate = hovertemplate

    fig.add_trace(heatmap)
    fig.update_layout(#title=_get_title(food_web),
                      width=width,
                      height=height,
                      autosize=True,
                      yaxis={'categoryarray': _get_array_order(graph, from_nodes),
                             'title': 'From' if not switch_axes else 'To'},
                      xaxis={'categoryarray': _get_array_order(graph, to_nodes, True),
                             'title': 'To' if not switch_axes else 'From'},
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          xanchor="right",
                          x=1,
                          y=1),
                      )
    fig.update_xaxes(showspikes=True, spikethickness=0.5)
    fig.update_yaxes(showspikes=True, spikesnap="cursor", spikemode="across", spikethickness=0.5)
    return fig



def draw_trophic_flows_heatmap(food_web, switch_axes=False, normalization='log', width=1200, height=800):
    '''Visualize flows between foodweb's trophic levels as a heatmap. The
    color at (x,y) represents the sum of flows from trophic level x to
    trophic level y.

    Parameters
    ----------
    food_web : foodwebs.FoodWeb
        Foodweb object.
    normalization : string, optional (default=log)
        Defines method of graph edges normalization.
        Available options are: 'diet', 'log', 'donor_control',
        'predator_control', 'mixed_control', 'linear' and 'tst'.
    switch_axes : bool, optional (default=False)
        If True, X axis will represent "to" trophic levels and Y - "from".
    width : int, optional (default=1200)
        Width of the plot.
    height : int, optional (default=800)
        Height of the plot.

    Returns
    -------
    heatmap : plotly.graph_objects.Figure
    '''
    if not switch_axes:
        hovertemplate = '%{y} --> %{x}: %{z:.3f}<extra></extra>'
    else:
        hovertemplate = '%{x} --> %{y}: %{z:.3f}<extra></extra>'

    tf_pd = _get_trophic_flows(food_web, normalization=normalization)
    heatmap = go.Heatmap(x=tf_pd['to' if not switch_axes else 'from'],
                         y=tf_pd['from' if not switch_axes else 'to'],
                         z=tf_pd['weights'],
                         xgap=0.2,
                         ygap=0.2,
                         colorscale=TROPHIC_LAYER_COLORS,
                         hoverongaps=False,
                         hovertemplate=hovertemplate)

    if normalization == 'log':
        z_orginal = _get_trophic_flows(food_web, normalization='linear').weights.values
        tickvals = range(int(np.log10(min(z_orginal))) + 1, int(np.log10(max(z_orginal))) + 1)
        ticktext = [10**x for x in range(len(tickvals))]
 
        heatmap.colorbar = dict(
            tick0=0,
            tickmode='array',
            tickvals=list(tickvals),
            ticktext=ticktext
        )
        heatmap.customdata = z_orginal
        if switch_axes:
            hovertemplate = '%{x} --> %{y}: %{customdata:.3f}<extra></extra>'
        else:
            hovertemplate = '%{y} --> %{x}: %{customdata:.3f}<extra></extra>'
        heatmap.hovertemplate = hovertemplate

    fig = go.Figure(data=heatmap)
    fig.update_layout(#title=_get_title(food_web),
                      width=width,
                      height=height,
                      autosize=True,
                      xaxis_showgrid=False,
                      yaxis_showgrid=False,
                      yaxis={'title': 'Trophic Layer From'if not switch_axes else 'Trophic Layer To',
                             'dtick': 1},
                      xaxis={'title': 'Trophic Layer To' if not switch_axes else 'Trophic Layer From',
                             'dtick': 1}
                      )
    return fig


def draw_trophic_flows_distribution(food_web, normalize=True, width=1000, height=800):
    '''Visualize flows between trophic levels as a stacked bar chart.

    Parameters
    ----------
    food_web : foodwebs.FoodWeb
        Foodweb object.
    normalize : bool, optional (default=True)
        If True, bars will represent percentages summing up to 100
    width : int, optional (default=600)
        Width of the plot.
    height : int, optional (default=800)
        Height of the plot.

    Returns
    -------
    heatmap : plotly.graph_objects.Figure
    '''
    tf_pd = _get_trophic_flows(food_web)
    tf_pd['to'] = tf_pd['to'].astype(str)

    if normalize:
        tf_pd['percentage'] = tf_pd['weights'] / tf_pd.groupby('from')['weights'].transform('sum')

    fig = px.bar(tf_pd,
                 y="from",
                 x="weights" if not normalize else "percentage",
                 color="to",
                 color_discrete_sequence=[x[1] for x in TROPHIC_LAYER_COLORS],
                 #title=_get_title(food_web),
                 height=width,
                 width=width,
                 template="simple_white",
                 hover_data={'from': ':d',
                             'to': ':d',
                             "weights" if not normalize else "percentage":  ':.4f'},
                 labels={
                     'from': 'Trophic Layer From',
                     'to': 'Trophic Layer To',
                     'percentage': 'Percentage of flow'},
                 orientation='h')
    return fig


def draw_network_for_nodes(food_web,
                           nodes=None,
                           file_name='food_web.html',
                           notebook=True,
                           height="800px",
                           width="100%",
                           no_flows_to_detritus=True,
                           **kwargs):
    '''Visualize subgraph of foodweb as a network.
    Parameters notebook, height, and width refer to initialization parameters of pyvis.network.Network.
    Additional parameters may be passed to hierachical repulsion layout as defined in
    pyvis.network.Network.hrepulsion. Examples are: node_distance, central_gravity,
    spring_length, or spring_strength.

    Parameters
    ----------
    food_web : foodwebs.FoodWeb
        Foodweb object.
    nodes : list of strings
        Nodes to include in subgraph to visualize.
    file_name : string, optional (default="food_web.html")
        File to save network (in html format)
    notebook - bool, optional (default=True)
        True if using jupyter notebook.
    height : string, optional (default="800px")
        Height of the canvas. See: pyvis.network.Network.hrepulsion
    width : string, optional (default="100%")
        Width of the canvas. See: pyvis.network.Network.hrepulsion

    Returns
    -------
    heatmap : pyvis.network.Network
    '''
    nt = Network(notebook=notebook,
                 height=height,
                 width=width,
                 directed=True,
                 layout=True,
                 heading='') #food_web.title)
    g = food_web.get_graph(mark_alive_nodes=True, no_flows_to_detritus=no_flows_to_detritus).copy()

    if not nodes:
        nodes = g.nodes()

    g = g.edge_subgraph([(x[0], x[1]) for x in g.edges() if x[0].replace(
        f'{fw.NOT_ALIVE_MARK} ', '') in nodes or x[1].replace(f'{fw.NOT_ALIVE_MARK} ', '') in nodes])

    a = {x: {'color': [x[1] for x in TROPHIC_LAYER_COLORS]
                      [int(decimal.Decimal(attrs['TrophicLevel']).to_integral_value())],
             'level': -attrs['TrophicLevel'],
             'title': f'''{x}<br> TrophicLevel: {attrs["TrophicLevel"]:.2f}
                                <br> Biomass: {attrs["Biomass"]:.2f}
                                <br> Import: {attrs["Import"]:.2f}
                                <br> Export: {attrs["Export"]:.2f}
                                <br> Respiration: {attrs["Respiration"]:.2f}''',
             'shape': 'box'} for x, attrs in g.nodes(data=True)}
    nx.set_node_attributes(g, a)

    # rename weight attribute to value
    nx.set_edge_attributes(g, {(edge[0], edge[1]): edge[2] for edge in g.edges(data='weight')}, 'value')

    nt.from_nx(g)
    nt.hrepulsion(node_distance=220, **kwargs)
    nt.set_edge_smooth('discrete')
    nt.show_buttons(filter_='physics')
    return nt.show(file_name)
