import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from collections import defaultdict


def _get_trophic_layer(graph, from_nodes, to_nodes):
    '''
    Creates thropic level Heatmap Trace
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
        colorscale='Teal',
        name='Trophic Layer',
        hoverinfo='skip'
    )


def _get_trophic_flows(net):
    ''' For each pair of trophic levels assigns sum of all weights in that pair '''
    graph = net.get_graph()

    trophic_flows = defaultdict(float)
    for n, n_trophic in set(graph.nodes(data='TrophicLevel')):
        for m, m_trophic in set(graph.nodes(data='TrophicLevel')):
            weight = graph.get_edge_data(n, m, default=0)
            if weight != 0:
                trophic_flows[(round(n_trophic), round(m_trophic))] += weight['weight']
    return pd.DataFrame([(x, y, w) for (x, y), w in trophic_flows.items()], columns=['from', 'to', 'weights'])


def _get_array_order(graph, nodes, reverse=False):
    def sort_key(x): return (x[1].get('TrophicLevel', 0), x[1].get('IsAlive', 0))
    return [x[0] for x in sorted(graph.nodes(data=True), key=sort_key, reverse=reverse) if x[0] in nodes]


def show_heatmap(food_web, normalization=None, show_trophic_layer=True, add_external_flows=False, switch_axes=False):
    '''
    Shows foodweb's heatmap

    Parameters
    ----------
    normalization - defines how flow weights should be normalized. See: prepare_data()
    show_trophic_layer : Bool - add background layer, showing 'from' node's trophic level
    add_external_flows : Bool - add columns for Imports, Exports, Respiration
    switch_axes: Bool - if False then from nodes are on y axis
    '''

    graph = food_web.get_graph(add_external_flows, mark_alive_nodes=True, normalization=normalization)
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
        colorscale='Emrld',  # 'Tealgrn',
        hoverongaps=False,
        hovertemplate='%{y} --> %{x}: %{z:.3f}<extra></extra>'
    )

    # fix color bar for log normalization
    if normalization == 'log':
        z_orginal = [x[2]['weight'] for x in food_web.get_graph(
            add_external_flows, mark_alive_nodes=True, normalization=None).edges(data=True)]
        ticktext = [10**x for x in range(int(math.log10(max(z_orginal))) + 1)]
        tickvals = range(int(math.log10(min(z_orginal))) + 1, int(math.log10(max(z_orginal))) + 1)

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
    fig.update_layout(title=food_web.title,
                      width=1200,
                      height=900,
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
    fig.show()


def draw_trophic_flows_heatmap(food_web, switch_axes=False, log_scale=False):
    '''
    Draws heatmap of flows between trophic levels
    '''
    if not switch_axes:
        hovertemplate = '%{y} --> %{x}: %{z:.3f}<extra></extra>'
    else:
        hovertemplate = '%{x} --> %{y}: %{z:.3f}<extra></extra>'

    tf_pd = _get_trophic_flows(food_web)
    z = np.log10(tf_pd['weights']) if log_scale else tf_pd['weights']

    heatmap = go.Heatmap(x=tf_pd['to' if not switch_axes else 'from'],
                         y=tf_pd['from' if not switch_axes else 'to'],
                         z=z,
                         xgap=0.2,
                         ygap=0.2,
                         colorscale='Emrld',  # 'Tealgrn',
                         hoverongaps=False,
                         hovertemplate=hovertemplate)

    if log_scale:
        ticktext = [10**x for x in range(int(math.log10(max(tf_pd['weights']))) + 1)]
        tickvals = range(int(math.log10(min(tf_pd['weights']))) + 1,
                         int(math.log10(max(tf_pd['weights']))) + 1)

        heatmap.colorbar = dict(
            tick0=0,
            tickmode='array',
            tickvals=list(tickvals),
            ticktext=ticktext
        )
        heatmap.customdata = tf_pd['weights']
        if switch_axes:
            hovertemplate = '%{x} --> %{y}: %{customdata:.3f}<extra></extra>'
        else:
            hovertemplate = '%{y} --> %{x}: %{customdata:.3f}<extra></extra>'
        heatmap.hovertemplate = hovertemplate

    fig = go.Figure(data=heatmap)
    fig.update_layout(title=food_web.title,
                      width=1200,
                      height=900,
                      autosize=True,
                      yaxis={'title': 'From' if not switch_axes else 'To',
                             'dtick': 1},
                      xaxis={'title': 'To' if not switch_axes else 'From',
                             'dtick': 1},
                      )
    fig.show()


def show_trophic_flows_distribution(net, normalize=False):
    tf_pd = _get_trophic_flows(net)
    tf_pd['to'] = tf_pd['to'].astype(str)

    if normalize:
        tf_pd['percentage'] = tf_pd['weights'] / tf_pd.groupby('from')['weights'].transform('sum')

    fig = px.bar(tf_pd,
                 y="from",
                 x="weights" if not normalize else "percentage",
                 color="to",
                 title=net.title,
                 height=600,
                 width=1000,
                 template="simple_white",
                 orientation='h')
    fig.show()
