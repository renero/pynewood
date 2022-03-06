import os
import string
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pydot
import pydotplus

from IPython.display import Image, display
from pydot import Dot
from typing import List

from .graph_utils import graph_to_adjacency, AnyGraph


def dot_graph(
    G: nx.DiGraph,
    undirected=False,
    plot: bool = True,
    name: str = "my_dotgraph",
    odots: bool = True,
    **kwargs,
) -> Dot:
    """
    Display a DOT of the graph in the notebook.

    Args:
        G (nx.Graph or DiGraph): the graph to be represented.
        undirected (bool): default False, indicates whether the plot is forced
            to contain no arrows.
        plot (bool): default is True, this flag can be used to simply generate
            the object but not plot, in case the object is needed to generate
            a PNG version of the DOT, for instance.
        name (str): the name to be embedded in the Dot object for this graph.
        odots (bool): represent edges with biconnections with circles (odots). if
            this is set to false, then the edge simply has no arrowheads.

    Returns:
        pydot.Dot object
    """
    if len(list(G.edges())) == 0:
        return None
    # Obtain the DOT version of the NX.DiGraph and visualize it.
    if undirected:
        G = G.to_undirected()
        dot_object = nx.nx_pydot.to_pydot(G)
    else:
        # Make a dot Object with edges reflecting biconnections as non-connected edges
        # or arrowheads as circles.
        dot_str = "strict digraph" + name + "{\nconcentrate=true;\n"
        for node in G.nodes():
            dot_str += f"{node};\n"
        if odots:
            options = "[arrowhead=odot, arrowtail=odot, dir=both]"
        else:
            options = "[dir=none]"
        for u, v in G.edges():
            if G.has_edge(v, u):
                dot_str += f"{u} -> {v} {options};\n"
            else:
                dot_str += f"{u} -> {v};\n"
        dot_str += "}\n"
        # src = graphviz.Source(dot_str)
        dot_object = pydotplus.graph_from_dot_data(dot_str)
        # old way of getting the dot_object
        # > dot_object = nx.nx_pydot.to_pydot(G)

    # This is to display single arrows with two heads instead of two arrows with
    # one head towards each direction.
    dot_object.set_concentrate(True)
    if plot:
        plot_dot(dot_object, **kwargs)

    return dot_object


def dot_graphs(
    dags: List[AnyGraph],
    dag_names: List[str] = None,
    num_rows=1,
    num_cols=2,
    undirected: bool = False,
    odots: bool = True,
    fig_size=(12, 8),
    **kwargs,
):
    """
    Make a plot with several Dots of the dags passed.
    Args:
        dags: A list of netowrkx graph objects
        dag_names: A list of names for the graphs passed
        num_rows: number of rows to use when creating subplots
        num_cols: number of cols to use when creating subplots
        undirected: whether the dot representation must be undirected (no arrows)
        odots: whether represent biconnections with circles in both directions.
        fig_size: tuple with the fig size for the plot.
        **kwargs: optional arguments to be passed to matplotlib

    Returns:
        None
    """
    pngs = []
    label = "".join(random.choice(string.ascii_lowercase) for _ in range(6))
    for d, dag in enumerate(dags):
        d_obj = dot_graph(dag, undirected=undirected, odots=odots, plot=False, **kwargs)
        output = f"./png/dag_{label}_{d}.png"
        if not os.path.exists("./png"):
            os.makedirs("./png")
        d_obj.write_png(output)
        pngs.append(output)

    # Represent all the DAGs together
    fig = plt.figure(figsize=fig_size)
    images = [plt.imread(png) for png in pngs]
    if dag_names is None:
        dag_names = [f"dag_{i}" for i in range(len(dags))]

    if len(dags) > num_cols and num_rows == 1:
        num_cols = len(dags)
    for i in range(num_rows * num_cols):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        if i >= len(pngs):  # empty image https://stackoverflow.com/a/30073252
            npArray = np.array([[[255, 255, 255, 255]]], dtype="uint8")
            ax.imshow(npArray, interpolation="nearest")
            ax.set_axis_off()
        else:
            ax.imshow(images[i])
            ax.set_title(f"{dag_names[i].upper()}")
            ax.set_axis_off()

    title = "Causal DAGs"
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_dot(dot_object: pydot.Dot, **kwargs) -> None:
    """ Displays a DOT object in the notebook """
    image = Image(dot_object.create_png(), **kwargs)
    display(image)


def dot_compared(g: AnyGraph, ref: AnyGraph, odots: bool = False) -> Dot:
    """
    Build a DOT object for the graph (g) taking into account a reference graph

    Arguments:
        g (Graph or DiGraph): this is the graph to be compared against the ground truth
        ref (Graph or DiGraph): this is the reference graph or ground truth
        odots (bool): whether represent bidirectional edges with circles

    Returns:
        Dot object

    """
    dot_string = """
    strict digraph  {
        concentrate=True;
    """
    dir_options = "arrowhead=odot, arrowtail=odot, dir=both" if odots else "dir=none"
    edge_weights = list(map(lambda t: t[2]['weight'], g.edges(data=True)))
    min_weight, max_weight = min(edge_weights), max(edge_weights)
    list(map(lambda t: t[2]['weight'], g.edges(data=True)))
    for u, v, data in g.edges(data=True):
        dot_string += f"{u:3s} -> {v:3s} "
        penwidth = int(data["weight"] - min_weight + 1)
        if ref.has_edge(u, v) and not g.has_edge(v, u):
            dot_string += f'[penwidth={penwidth}, color="darkgreen"];\n'
        elif g.has_edge(v, u):
            dot_string += f'[penwidth={penwidth}, color="darkgreen", '
            dot_string += f'style="dashed", {dir_options}];\n'
        elif ref.has_edge(v, u):
            dot_string += f'[penwidth={penwidth}, color="red"];\n'
        else:
            dot_string += f'[style="dashed", color="darkgrey"];\n'
    dot_string += "}"
    return pydot.graph_from_dot_data(dot_string)[0]


def dot_reference(ref: AnyGraph, g: AnyGraph) -> Dot:
    """
    Build a DOT object for the reference graph, highlighting matches obtained in the
    other graph.

    Arguments:
        g (Graph or DiGraph): this is the graph obtained against the ground truth
        ref (Graph or DiGraph): this is the reference graph or ground truth to be
        formatted

    Returns:
        Dot object

    """
    dot_string = """
    strict digraph  {
        concentrate=True;
    """
    list(map(lambda t: t[2]['weight'], g.edges(data=True)))
    for u, v, data in ref.edges(data=True):
        dot_string += f"{u:3s} -> {v:3s} "
        if g.has_edge(u, v) and not g.has_edge(v, u):
            dot_string += f'[color="darkgreen"];\n'
        elif g.has_edge(v, u):
            dot_string += f'[style="dashed", color="green"];\n'
        else:
            dot_string += ';\n'
    dot_string += "}"
    return pydot.graph_from_dot_data(dot_string)[0]


def dot_comparison(
        dag: AnyGraph,
        reference: AnyGraph,
        dag_names: List[str] = None,
        odots: bool = True,
        **kwargs,
):
    """
    Make a plot with several Dots of the dags passed.
    Args:
        dag: The graph to be compared.
        reference: The graph used as reference
        dag_names: A list of names for the graphs passed
        odots: Whether representing bidirectional edges with circles
        **kwargs: optional arguments to be passed to matplotlib

    Returns:
        None
    """
    figsize = kwargs.get("figsize", (12, 8))
    title = kwargs.get("title", "Causal DAGs comparison")
    pngs = []
    if not os.path.exists("./png"):
        os.makedirs("./png")

    # Build the DOT for the compared graph
    d_obj = dot_compared(dag, reference, odots)
    output = f"./png/dag_comp.png"
    d_obj.write_png(output)
    pngs.append(output)

    # Build the DOT for the reference graph (ground truth)
    d_obj = dot_reference(reference, dag)
    output = f"./png/dag_ref.png"
    d_obj.write_png(output)
    pngs.append(output)

    # Represent all the DAGs together
    fig = plt.figure(figsize=figsize)
    images = [plt.imread(png) for png in pngs]
    if dag_names is None:
        dag_names = ["DAG", "REFERENCE"]

    for i in range(2):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.imshow(images[i])
        ax.set_title(f"{dag_names[i].upper()}")
        ax.set_axis_off()

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_graph(graph: nx.DiGraph) -> None:
    """Plot a graph using default Matplotlib methods"""
    pos = nx.circular_layout(graph, scale=20)
    nx.draw(graph, pos,
            nodelist=graph.nodes(),
            node_color="lightblue",
            node_size=800,
            width=2,
            alpha=0.9,
            with_labels=True)
    plt.show()


def plot_graphs(G: nx.MultiDiGraph, H: nx.DiGraph) -> None:
    """Plot two graphs side by side."""
    pos1 = nx.circular_layout(G, scale=20)
    pos2 = nx.circular_layout(H, scale=20)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax = axes.flatten()
    nx.draw_networkx(G, pos1, node_color="lightblue",
                     node_size=800, edge_color='orange',
                     width=2, alpha=0.9, ax=ax[0])
    ax[0].set_axis_off()
    ax[0].set_title("Ground Truth")
    nx.draw_networkx(H, pos2, node_color="lightblue",
                     node_size=800, edge_color='lightblue',
                     width=2, alpha=0.9, ax=ax[1])
    ax[1].set_axis_off()
    ax[1].set_title("Other")
    plt.tight_layout()
    plt.show()


def plot_compared_graph(G: nx.DiGraph, H: nx.DiGraph) -> None:
    """
    Iterate over the composed graph's edges and nodes, and assign to these a
    color depending on which graph they belong to (including both at the same
    time too). This could also be extended to adding some attribute indicating
    to which graph it belongs too.
    Intersecting nodes and edges will have a magenta color. Otherwise they'll
    be green or blue if they belong to the G or H Graph respectively
    """
    GH = nx.compose(G, H)
    # set edge colors
    edge_colors = dict()
    for edge in GH.edges():
        if G.has_edge(*edge):
            if H.has_edge(*edge):
                edge_colors[edge] = 'black'
                continue
            edge_colors[edge] = 'lightgreen'
        elif H.has_edge(*edge):
            edge_colors[edge] = 'orange'

    # set node colors
    G_nodes = set(G.nodes())
    H_nodes = set(H.nodes())
    node_colors = []
    for node in GH.nodes():
        if node in G_nodes:
            if node in H_nodes:
                node_colors.append('lightgrey')
                continue
            node_colors.append('lightgreen')
        if node in H_nodes:
            node_colors.append('orange')

    pos = nx.circular_layout(GH, scale=20)
    nx.draw(GH, pos,
            nodelist=GH.nodes(),
            node_color=node_colors,
            edgelist=edge_colors.keys(),
            edge_color=edge_colors.values(),
            node_size=800,
            width=2, alpha=0.5,
            with_labels=True)


def plot_adjacency(g: nx.Graph, ax=None):
    """
    Plots the adjacency matrix as explained by scikit contributor
    Andreas Mueller in Columbia lectures, ordering and grouping
    (numerical) features with higher correlation.

    Returns:
        None
    """
    mat = graph_to_adjacency(g)
    features = sorted(list(g.nodes))
    num_features = len(features)

    if ax is None:
        _, ax = plt.subplots()
    plt.xticks(fontsize=10)
    ax.set_title("Grouped Adjacency Matrix")
    ax.matshow(mat, interpolation="nearest")
    for (j, i), label in np.ndenumerate(mat):
        ax.text(i, j, f"{label:.2g}", ha="center", va="center")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticks(range(num_features))
    ax.set_xticklabels(features)
    ax.set_yticks(range(num_features))
    ax.set_yticklabels(features)
