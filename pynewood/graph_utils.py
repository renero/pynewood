"""
This module incorporates util functions for graphs.
"""
from pathlib import Path

from deprecated import deprecated
from typing import List, Union, Tuple, Dict, Callable, Set

import networkx as nx
import numpy
import numpy as np
import pandas as pd
import pydot as pydot
import pydotplus
from networkx import Graph, DiGraph

from . file_utils import file_exists

AnyGraph = Union[nx.Graph, nx.DiGraph]


@deprecated(version="0.234", reason="Use `compare_graphs` instead.")
def compute_graph_metrics(truth, result):
    """
    Compute graph precision and recall. Recall refers to the list of edges
    that have been correctly identified in result, and precision, to the
    ratio of edges that correctly math to those in the ground truth.

    Arguments:
        truth: A list of edges representing the true structure of the graph
               to compare with.
        result: The dag for which to measure the metrics.

    Returns:
        precision, recall values as floats

    Example:
        >>> dag1 = [('a', 'b'), ('a', 'c'), ('c', 'd'), ('c', 'b')]
        >>> dag2 = [('a', 'b'), ('a', 'c'), ('b', 'd')]
        >>> prec, rec = compute_graph_metrics(dag1, dag2)
        >>> print(prec, rec)
        >>> 0.75 0.5

    """
    # Convert the ground truth and target into a set of tuples with edges
    if not isinstance(truth, set):
        ground_truth = set([tuple(pair) for pair in truth])
    elif isinstance(truth, set):
        ground_truth = truth
    else:
        raise TypeError("Truth argument must be a list or a set.")
    if not isinstance(result, set):
        target = set([tuple(pair) for pair in result])
    elif isinstance(result, set):
        target = result
    else:
        raise TypeError("Results argument must be a list or a set.")

    # Set the total number of edges if ground truth skeleton
    total = max([float(len(ground_truth)), float(len(target))])
    true_positives = len(ground_truth.intersection(target))
    false_positives = len(target - ground_truth.intersection(target))
    precision = 1. - (false_positives / total)
    recall = true_positives / total

    return precision, recall


def build_graph(list_nodes: List, matrix: np.ndarray,
                threshold=0.05, zero_diag=True) -> nx.Graph:
    """
    Builds a graph from an adjacency matrix. For each position i, j, if the
    value is greater than the threshold, an edge is added to the graph. The
    names of the vertices are in the list of nodes pased as argument, whose
    order must match the columns in the matrix.

    The diagonal of the matrix is set to zero to avoid inner edges, but this
    behavior can be overridden by setting zero_diag to False.

    Args:
        list_nodes: a list with the names of the graph's nodes.
        matrix: a numpy ndarray with the weights to be used
        threshold: the threshold above which a vertex is created in the graph
        zero_diag: boolean indicating whether zeroing the diagonal. Def True.

    Returns:
        nx.Graph: A graph with edges between values > threshold.

    Example:
        >>> matrix = np.array([[0., 0.3, 0.2],[0.3, 0., 0.2], [0.0, 0.2, 0.]])
        >>> dag = build_graph(['a','b','c'], matrix, threshold=0.1)
        >>> dag.edges()
            EdgeView([('a', 'b'), ('a', 'c'), ('b', 'c')])
    """
    M = np.copy(matrix)
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix must be square")
    if M.shape[1] != len(list_nodes):
        raise ValueError("List of nodes doesn't match number of rows/cols")
    if zero_diag:
        np.fill_diagonal(M, 0.)
    graph = nx.Graph()
    for (i, j), x in np.ndenumerate(M):
        if M[i, j] > threshold:
            graph.add_edge(list_nodes[i], list_nodes[j],
                           weight=M[i, j])
    for node in list_nodes:
        if node not in graph.nodes():
            graph.add_node(node)
    return graph


@deprecated(version="0.2.34", reason="Use graph_print_edges()")
def print_graph_edges(graph: nx.Graph):
    graph_print_edges(graph)


def graph_print_edges(graph: nx.Graph):
    """
    Pretty print the nodes of a graph, with weights

    Args:
         graph: the graph to be printed out.
    Returns:
        None.
    Example:
        >>> matrix = np.array([[0., 0.3, 0.2],[0.3, 0., 0.2], [0.0, 0.2, 0.]])
        >>> dag = build_graph(['a','b','c'], matrix, threshold=0.1)
        >>> graph_print_edges(dag)
            Graph contains 3 edges.
            a –– b +0.3000
            a –– c +0.2000
            b –– c +0.2000

    """
    mx = max([len(s) for s in list(graph.nodes)])
    edges = list(graph.edges)
    print(f'Graph contains {len(edges)} edges.')

    # Check if this graph contain weight information
    get_edges = getattr(graph, "edges", None)
    if callable(get_edges):
        edges_weights = get_edges(data='weight')
    else:
        edges_weights = edges

    # Printout
    for edge in edges_weights:
        if len(edge) == 3 and edge[2] is not None:
            print(("{:" + str(mx) + "s} –– {:" + str(mx) + "s} {:+.4f}").format(
                edge[0], edge[1], edge[2]))
        else:
            print(("{:" + str(mx) + "s} –– {:" + str(mx) + "s}").format(
                edge[0], edge[1]))


def graph_to_adjacency(graph: AnyGraph, weight_label: str = "weight") -> numpy.ndarray:
    """
    A method to generate the adjacency matrix of the graph. Labels are
    sorted for better readability.

    Args:
        graph: (Union[Graph, DiGraph]) the graph to be converted.
        weight_label: the label used to identify the weights.

    Return:
        graph: (numpy.ndarray) A 2d array containing the adjacency matrix of
            the graph.
    """
    symbol_map = {"o": 1, ">": 2, "-": 3}
    labels = sorted(list(graph.nodes))  # [node for node in self]
    mat = np.zeros((len(labels), (len(labels))))
    for x in labels:
        for y in labels:
            if graph.has_edge(x, y):
                if bool(graph.get_edge_data(x, y)):
                    if y in graph.get_edge_data(x, y).keys():
                        mat[labels.index(x)][labels.index(y)] = symbol_map[
                            graph.get_edge_data(x, y)[y]
                        ]
                    else:
                        mat[labels.index(x)][labels.index(y)] = graph.get_edge_data(
                            x, y
                        )[weight_label]
                else:
                    mat[labels.index(x)][labels.index(y)] = 1
    return mat


def graph_from_adjacency(
        adjacency: np.ndarray, node_labels=None, th=None, inverse: bool = False,
        absolute_values: bool = False
) -> nx.DiGraph:
    """
    Manually parse the adj matrix to shape a dot graph

    Args:
        adjacency: a numpy adjacency matrix
        node_labels: an array of same length as nr of columns in the adjacency
            matrix containing the labels to use with every node.
        th: (float) weight threshold to be considered a valid edge.
        inverse (bool): Set to true if rows in adjacency reflects where edges are
            comming from, instead of where are they going to.
        absolute_values: Take absolute value of weight label to check if its greater
            than the threshold.

    Returns:
         The Graph (DiGraph)
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(adjacency.shape[1]))

    # What to do with absolute values?
    not_abs = lambda x: x
    w_val = np.abs if absolute_values else not_abs
    weight_gt = lambda w, thresh: w != 0.0 if thresh is None else w_val(w) > thresh

    # A method to check if weight is greater than threshold, only if has been specified
    # def check_weight(w_val, threshold):
    #     if threshold is None:
    #         return True
    #     return weight(w_val) > threshold

    # Do I have a threshold to consider?
    for i, row in enumerate(adjacency):
        for j, value in enumerate(row):
            if inverse:
                if weight_gt(adjacency[j][i], th):
                    G.add_edge(i, j, weight=w_val(adjacency[j][i]))
            else:
                if weight_gt(value, th):
                    G.add_edge(i, j, weight=w_val(value))  # , arrowhead="normal")
    # Map the current column numbers to the letters used in toy dataset
    if node_labels is not None and len(node_labels) == adjacency.shape[1]:
        mapping = dict(zip(sorted(G), node_labels))
        G = nx.relabel_nodes(G, mapping)

    return G


def graph_from_adjacency_file(file: Union[Path, str], th=0.0) -> Tuple[
    nx.DiGraph, pd.DataFrame]:
    """
    Read Adjacency matrix from a file and return a Graph

    Args:
        file: (str) the full path of the file to read
        th: (float) weight threshold to be considered a valid edge.
    Returns:
        DiGraph, DataFrame
    """
    df = pd.read_csv(file, dtype="str")
    df = df.astype("float64")
    labels = list(df)
    G = graph_from_adjacency(df.values, node_labels=labels, th=th)
    return G, df


def graph_to_adjacency_file(graph: AnyGraph, output_file: Union[Path, str]):
    """
    A method to write the adjacency matrix of the graph to a file. If graph has
    weights, these are the values stored in the adjacency matrix.

    Args:
        graph: (Union[Graph, DiGraph] the graph to be saved
        output_file: (str) The full path where graph is to be saved
    """
    mat = graph_to_adjacency(graph)
    labels = sorted(list(graph.nodes))
    f = open(output_file, "w")
    f.write(",".join([f"{label}" for label in labels]))
    f.write("\n")
    for i in range(len(labels)):
        f.write(f"{labels[i]}")
        f.write(",")
        f.write(",".join([str(point) for point in mat[i]]))
        f.write("\n")
    f.close()


def graph_from_dot_file(dot_file: Union[str, Path]) -> nx.DiGraph:
    """ Returns a NetworkX DiGraph from a DOT FILE. """
    dot_object = pydot.graph_from_dot_file(dot_file)
    dotplus = pydotplus.graph_from_dot_data(dot_object[0].to_string())
    dotplus.set_strict(True)
    return nx.nx_pydot.from_pydot(dotplus)


def graph_from_dot(dot_object: pydot.Dot) -> nx.DiGraph:
    """ Returns a NetworkX DiGraph from a DOT object. """
    dotplus = pydotplus.graph_from_dot_data(dot_object.to_string())
    dotplus.set_strict(True)
    return nx.nx_pydot.from_pydot(dotplus)


def graph_to_dot(g: AnyGraph) -> pydot.Dot:
    """Converts a graph into a dot structure"""
    return nx.drawing.nx_pydot.to_pydot(g)


def graph_to_dot_file(g: AnyGraph, location: Union[Path, str]) -> None:
    """ Converts graph into a pyDot object and saves it to specified location"""
    nx.drawing.nx_pydot.write_dot(g, location)


def graph_fom_csv(
        graph_file: Union[Path, str],
        graph_type: Callable,
        source_label="from",
        target_label="to",
        edge_attr_label=None,
):
    """
    Read Graph from a CSV file with "FROM", "TO" and "WEIGHT" fields

    Args:
        graph_file: a full path with the filename
        graph_type: Graph or DiGraph
        source_label: name of the "from"/cause column in the dataset
        target_label: name of the "to"/effect column in the dataset
        edge_attr_label: name of the weight, if any (def: None)

    Returns:
        networkx.Graph or networkx.DiGraph
    """
    edges = pd.read_csv(graph_file)
    Graphtype = graph_type()
    ugraph = nx.from_pandas_edgelist(
        edges,
        source=source_label,
        target=target_label,
        edge_attr=edge_attr_label,
        create_using=Graphtype,
    )
    return ugraph


def graph_to_csv(graph: AnyGraph, output_file: Union[Path, str]):
    """
    Save a GrAPH to CSV file with "FROM", "TO" and "CSV"
    """
    if file_exists(output_file, "."):
        output_file = f"New_{output_file}"
    skeleton = pd.DataFrame(list(graph.edges(data='weight')))
    skeleton.columns = ['from', 'to', 'weight']
    skeleton.to_csv(output_file, index=False)


def graph_weights(graph: AnyGraph, field="weight"):
    """
    Returns graph weights, or the name of the data field for each edge in the graph.

    Args:
        graph (Graph or DiGraph): the graph from wich to extract the field values
        field (str): The name of the field for which to extract the values. By
            default it is 'weight'.

    Returns:
        Numpy.array with the values of the specified field.
    """
    return np.array([
        data[field] if field in graph[s][t] else 0.0
        for s, t, data in graph.edges(data=True)
    ])


def graph_filter(graph: Union[Path, str],
                 threshold,
                 field="weight",
                 lower: bool = False):
    """
    Filter a graph taking only those edges whose weight is > threshold

    Args:
        graph: The graph to be filtered
        threshold: The minimum value to act as filter for edges weight
        field (str): The name of the weight to use as filter. Default is weight.
        lower (bool): If True the method returns edges whose weight is < threshold.

    Returns:
        A graph (same type as original) with only the edges filtered.
    """
    GType = type(graph)
    ng = GType()
    ng.add_nodes_from(graph)
    for u, v, d in graph.edges(data=True):
        comparison = d[field] < threshold if lower else d[field] >= threshold
        if comparison:
            ng.add_edge(u, v, weight=d[field])
    return ng


def graph_from_parent_ids(
        parents_list: Dict[int, List[int]], node_names: List[str]
) -> nx.DiGraph:
    """
    Build a graph from a list of parent ids. Each key in the dict is the id number
    of a node whose parents are in the values for that key.

    Example: {3: [], 0: [3], 2: [3, 0], 4: [3, 0, 2], 1: [3, 0, 2, 4]}

    The node "3" has no parents, the parent of "0" is "3", and so on.

    Arguments:
         parents_list (Dict[int, List[int]]): a dictionary with nodes as keys and
            lists of parents for each node as list of values.
        node_names (List[str]): The names of the nodes.

    Returns:
        A directed graph representing the hierarchy represented by the list of parents
    """
    g = nx.DiGraph()
    g.add_nodes_from(node_names)
    for child in parents_list.keys():
        parents = parents_list[child]
        if not parents:
            continue
        for parent in parents:
            g.add_edge(node_names[parent], node_names[child])

    return g


def graph_from_dictionary(d: Dict[str, Union[str, List[str]]]) -> AnyGraph:
    g = nx.DiGraph()
    for node, parents in d.items():
        for parent in parents:
            g.add_edge(parent, node)
    return g


def graph_union(graphs: List[AnyGraph], nodes: List[Union[str, int]]):
    """
    Computes the intersection of several graphs as the graph with the edges
    in common among all them. The resulting edges' weights are the nr of times
    that they are present in the set.

    Args:
        graphs: (List[nx.Graph] or List[nx.DiGraph]) A list of graphs
        nodes: Default list of nodes for the resulting graph, to ensure that
            graph is populated with at least these nodes, even though not all
            edges link them entirely

    Returns:
        nx.Digraph with the edges in common, weighted by the nr of times they appear
    """
    assert len(graphs) > 1, \
        "This method needs more than one graph to compute intersection"
    G = nx.DiGraph()
    if nodes is not None:
        G.add_nodes_from(nodes)
    for g in graphs:
        for u, v, d in g.edges(data=True):
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
                continue
            G.add_edge(u, v, weight=1)

    return G


def graph_biconnections(g) -> Set[Tuple[str, str]]:
    """
    Returns all bidirectional connections in a graph.
    Args:
        g: a networkx graph

    Returns:
        A set with the pairs of nodes bidirectionally connected.
    """
    def have_bidirectional_relationship(G, node1, node2):
        return G.has_edge(node1, node2) and G.has_edge(node2, node1)

    biconnections = set()
    for u, v in g.edges():
        if u > v:  # Avoid duplicates, such as (1, 2) and (2, 1)
            v, u = u, v
        if have_bidirectional_relationship(g, u, v):
            biconnections.add((u, v))

    return biconnections
