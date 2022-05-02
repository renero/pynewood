import networkx as nx
from .s_c_c import filter_big_scc, get_big_sccs, nodes_in_scc
from .helper_funs import pick_from_dict
from .remove_self_loops import remove_self_loops_from_graph

def remove_cycle_edges_by_ranking_score_iterately(sccs, players, edges_to_be_removed, is_Forward):
    while True:
        graph = sccs.pop()
        node_scores_dict = {}
        for node in graph.nodes():
            node_scores_dict[node] = players[node]
        max_k, max_v, min_k, min_v = pick_from_dict(node_scores_dict, "both")

        if is_Forward:
            node, score = max_k, max_v
            target_edges = [(node, v) for v in graph.successors(node)]
            #target_edges = [(v,node) for v in graph.predecessors_iter(node)]
        else:
            node, score = min_k, min_v
            target_edges = [(v, node) for v in graph.predecessors(node)]

        edges_to_be_removed += target_edges
        # Graph is frozen and cannot be modified
        # graph.remove_edges_from(target_edges)
        unfrozen_graph = nx.DiGraph(graph)
        unfrozen_graph.remove_edges_from(target_edges)
        # sub_graphs = filter_big_scc(graph,target_edges)
        sub_graphs = filter_big_scc(unfrozen_graph, target_edges)
        if sub_graphs:
            sccs += sub_graphs
        if not sccs:
            return


def scores_of_nodes_in_scc(sccs, players):
    scc_nodes = nodes_in_scc(sccs)
    scc_nodes_score_dict = {}
    for node in scc_nodes:
        scc_nodes_score_dict[node] = players[node]
    # print("# scores of nodes in scc: %d" % (len(scc_nodes_score_dict)))
    return scc_nodes_score_dict


def scc_based_to_remove_cycle_edges_iterately(g, nodes_score, is_Forward):
    big_sccs = get_big_sccs(g)
    if len(big_sccs) == 0:
        print("After removal of self loop edgs: %s" %
              nx.is_directed_acyclic_graph(g))
        return []
    scc_nodes_score_dict = scores_of_nodes_in_scc(big_sccs, nodes_score)
    edges_to_be_removed = []
    remove_cycle_edges_by_ranking_score_iterately(
        big_sccs, scc_nodes_score_dict, edges_to_be_removed, is_Forward)
    # print(" # edges to be removed: %d" % len(edges_to_be_removed))
    return edges_to_be_removed


def remove_cycle_edges_BF_iterately(g, players, is_Forward=True, score_name="socialagony"):
    self_loops = remove_self_loops_from_graph(g)
    edges_to_be_removed = scc_based_to_remove_cycle_edges_iterately(
        g, players, is_Forward)
    edges_to_be_removed = list(set(edges_to_be_removed))
    return edges_to_be_removed+self_loops
