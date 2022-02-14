import numpy as np
import pytest

from ..graph_utils import compute_graph_metrics, build_graph


def test_compute_graph_metrics():
    dag1 = [('a', 'b'), ('a', 'c'), ('c', 'd'), ('c', 'b')]
    dag2 = [('a', 'b'), ('a', 'c'), ('b', 'd')]
    prec, rec = compute_graph_metrics(dag1, dag2)
    if prec != 0.75:
        pytest.fail('Wrong precision')
    if rec != 0.5:
        pytest.fail('Wrong precision')

    with pytest.raises(TypeError):
        compute_graph_metrics(0.0, 0.0)


def test_build_graph():
    matrix = np.array([[0., 0.3, 0.2], [0.3, 0., 0.2], [0.0, 0.2, 0.]])
    dag = build_graph(['a', 'b', 'c'], matrix, threshold=0.1)
    e = list(dag.edges())
    if e[0] != ('a','b') or e[1] != ('a','c') or e[2] != ('b','c'):
        pytest.fail("Wrong computation of graph")

    with pytest.raises(ValueError):
        build_graph(['a', 'b', 'c'], matrix[1:, :])

    with pytest.raises(ValueError):
        build_graph(['a', 'b'], matrix)
