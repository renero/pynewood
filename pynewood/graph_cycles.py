import argparse
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
    graph_file, nodes_score_dict, score_name="socialagony", nodetype=int
):

    g = nx.read_edgelist(
        graph_file, create_using=nx.DiGraph(), nodetype=nodetype)
    # greedy
    e1 = scc_based_to_remove_cycle_edges_iterately(g, nodes_score_dict)
    g = nx.read_edgelist(
        graph_file, create_using=nx.DiGraph(), nodetype=nodetype)
    # forward
    e2 = remove_cycle_edges_BF_iterately(
        g, nodes_score_dict, is_Forward=True, score_name=score_name
    )
    # backward
    g = nx.read_edgelist(
        graph_file, create_using=nx.DiGraph(), nodetype=nodetype)
    e3 = remove_cycle_edges_BF_iterately(
        g, nodes_score_dict, is_Forward=False, score_name=score_name
    )
    return e1, e2, e3


def remove_cycle_edges_by_voting(graph_file, set_edges_list, nodetype=int):
    edges_score = get_edges_voting_scores(set_edges_list)
    e = remove_cycle_edges_heuristic(
        graph_file, edges_score, nodetype=nodetype)
    return e


def remove_cycle_edges_by_hierarchy(
    graph_file, nodes_score_dict, score_name="socialagony", nodetype=int
):
    e1, e2, e3 = remove_cycle_edges_strategies(
        graph_file, nodes_score_dict, score_name=score_name, nodetype=nodetype
    )
    e4 = remove_cycle_edges_by_voting(
        graph_file, [set(e1), set(e2), set(e3)], nodetype=nodetype
    )
    return e1, e2, e3, e4


def computing_hierarchy(graph_file, players_score_func_name, nodetype=int):
    g = nx.read_edgelist(
        graph_file, create_using=nx.DiGraph(), nodetype=nodetype)
    print("start computing trueskill...")
    players = graphbased_trueskill(g)

    return players


def breaking_cycles_by_hierarchy_performance(
    graph_file, gt_file, players_score_name="trueskill", nodetype=int
):
    players_score_dict = computing_hierarchy(
        graph_file, players_score_name, nodetype=nodetype
    )
    e1, e2, e3, e4 = remove_cycle_edges_by_hierarchy(
        graph_file, players_score_dict, players_score_name, nodetype=nodetype
    )
    print(f"TS_G,    removes: {len(e1)} edges; {', '.join([str(t) for t in e1])}")
    print(f"TS_F,    removes: {len(e1)} edges; {', '.join([str(t) for t in e1])}")
    print(f"TS_B,    removes: {len(e1)} edges; {', '.join([str(t) for t in e1])}")
    print(f"TS_Vote, removes: {len(e1)} edges; {', '.join([str(t) for t in e1])}")

    return e1, e2, e3, e4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--graph_file", default=" ",
                        help="input graph file name (edges list)")
    parser.add_argument("-t", "--gt_edges_file", default=None,
                        help="ground truth edges file")

    args = parser.parse_args()
    graph_file = args.graph_file
    gt_file = args.gt_edges_file

    graph = nx.read_edgelist(
        graph_file, create_using=nx.DiGraph(), nodetype=int)
    n_cycles = len(list(nx.simple_cycles(graph)))
    print(f"Cycles ({n_cycles}):")
    for cycle in nx.simple_cycles(graph):
        print(cycle)

    e1, e2, e3, e4 = breaking_cycles_by_hierarchy_performance(
        graph_file, gt_file)
    for u, v in e4:
        graph.remove_edge(u, v)

    n_cycles = len(list(nx.simple_cycles(graph)))
    print(f"Cycles after removal: ({n_cycles})")
    for cycle in nx.simple_cycles(graph):
        print(cycle)
