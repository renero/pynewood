import networkx as nx

from scc.remove_cycles_by_hierarchy_greedy import scc_based_to_remove_cycle_edges_iterately
from scc.remove_cycles_by_hierarchy_BF import remove_cycle_edges_BF_iterately
from scc.remove_cycles_by_hierarchy_voting import remove_cycle_edges_heuristic
from scc.true_skill import graphbased_trueskill


def get_edges_voting_scores(set_edges_list):
    total_edges = set()
    for edges in set_edges_list:
        total_edges = total_edges | edges
    edges_score = {}
    for e in total_edges:
        edges_score[e] = len(list(filter(lambda x: e in x, set_edges_list)))
    return edges_score


def remove_cycle_edges_strategies(
    graph, nodes_score_dict, score_name="trueskill"
):
    g = graph.copy()
    # greedy
    e1 = scc_based_to_remove_cycle_edges_iterately(g, nodes_score_dict)
    g = graph.copy()
    # forward
    e2 = remove_cycle_edges_BF_iterately(
        g, nodes_score_dict, is_Forward=True, score_name=score_name
    )
    # backward
    g = graph.copy()
    e3 = remove_cycle_edges_BF_iterately(
        g, nodes_score_dict, is_Forward=False, score_name=score_name
    )
    return e1, e2, e3


def remove_cycle_edges_by_voting(graph, set_edges_list, nodetype=int):
    edges_score = get_edges_voting_scores(set_edges_list)
    e = remove_cycle_edges_heuristic(graph, edges_score)
    return e


def remove_cycle_edges_by_hierarchy(graph: nx.DiGraph, nodes_score_dict, verbose=False):
    e1, e2, e3 = remove_cycle_edges_strategies(
        graph, nodes_score_dict, score_name="trueskill"
    )
    e4 = remove_cycle_edges_by_voting(
        graph, [set(e1), set(e2), set(e3)]
    )
    return e1, e2, e3, e4


def break_cycles(graph: nx.DiGraph, verbose=False):
    if verbose:
        print("start computing trueskill...")
    players_score_dict = graphbased_trueskill(graph, verbose=verbose)

    e1, e2, e3, e4 = remove_cycle_edges_by_hierarchy(
        graph, players_score_dict, verbose=verbose)

    if verbose:
        print(
            f"TS_G,    removes: {len(e1)} edges; {', '.join([str(t) for t in e1])}")
        print(
            f"TS_F,    removes: {len(e1)} edges; {', '.join([str(t) for t in e1])}")
        print(
            f"TS_B,    removes: {len(e1)} edges; {', '.join([str(t) for t in e1])}")
        print(
            f"TS_Vote, removes: {len(e1)} edges; {', '.join([str(t) for t in e1])}")

    return e1, e2, e3, e4


if __name__ == "__main__":
    graph_file = "/Users/renero/phd/code/breaking_cycles_in_noisy_hierarchies/mydata/g_rfe.edges"
    graph = nx.read_edgelist(
        graph_file, create_using=nx.DiGraph(), nodetype=int)
    n_cycles = len(list(nx.simple_cycles(graph)))
    print(f"Cycles ({n_cycles}):")
    for cycle in nx.simple_cycles(graph):
        print(cycle)

    e1, e2, e3, e4 = break_cycles(graph, verbose=True)
    for u, v in e4:
        graph.remove_edge(u, v)

    n_cycles = len(list(nx.simple_cycles(graph)))
    print(f"Cycles after removal: ({n_cycles})")
    for cycle in nx.simple_cycles(graph):
        print(cycle)
