import argparse
from trueskill import Rating, quality_1vs1, rate_1vs1
import networkx as nx
import numpy as np
import time
from datetime import datetime
import random

from .measures import measure_pairs_agreement
from .s_c_c import scc_nodes_edges


def compute_trueskill(pairs, players, verbose=False):
    if not players:
        for u, v in pairs:
            if u not in players:
                players[u] = Rating()
            if v not in players:
                players[v] = Rating()

    # start = time.time()
    random.shuffle(pairs)
    for u, v in pairs:
        players[v], players[u] = rate_1vs1(players[v], players[u])

    end = time.time()
    # if verbose:
    #     print("time used in computing true skill (per iteration): %0.4f s" %
    #         (end - start))
    return players


def get_players_score(players, n_sigma):
    relative_score = {}
    for k, v in players.items():
        relative_score[k] = players[k].mu - n_sigma * players[k].sigma
    return relative_score


def trueskill_ratings(pairs, iter_times=15, n_sigma=3, threshold=0.85, verbose=False):
    # start = datetime.now()
    players = {}
    for i in range(iter_times):
        players = compute_trueskill(pairs, players, verbose=verbose)
        relative_scores = get_players_score(players, n_sigma=n_sigma)
        accu = measure_pairs_agreement(pairs, relative_scores)
        if accu >= threshold:
            return relative_scores
    end = datetime.now()
    # time_used = end - start
    # if verbose:
    #     print("time used in computing true skill: %0.4f s, iteration time is: %i" %
    #         ((time_used.seconds), (i+1)))
    return relative_scores


def graphbased_trueskill(g, iter_times=15, n_sigma=3, threshold=0.95, verbose=False):
    relative_scores = trueskill_ratings(
        list(g.edges()), iter_times=iter_times, n_sigma=n_sigma, threshold=threshold, verbose=verbose)
    scc_nodes, scc_edges, nonscc_nodes, nonscc_edges = scc_nodes_edges(g)
    # if verbose:
    #     print("----scc-------")
    scc_accu = measure_pairs_agreement(scc_edges, relative_scores)
    # if verbose:
    #     print("----non-scc---")
    nonscc_accu = measure_pairs_agreement(nonscc_edges, relative_scores)
    if verbose:
        print("scc accu: %0.4f, nonscc accu: %0.4f" % (scc_accu, nonscc_accu))
    return relative_scores
