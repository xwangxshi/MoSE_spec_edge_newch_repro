import itertools
import os
import unittest

import dill
import networkx as nx
import numpy as np
import pandas as pd

from pact.naive_exec import naive_pandas_plan_exec
from pact.two_root import build_two_root_plan, execute_two_root_plan


HERE = os.path.dirname(os.path.abspath(__file__))
BASIS_DIR = os.path.join(HERE, '..', 'bases', 'cycles')


def brute_force_two_root(pattern, host, root, second_root):
    pattern_nodes = list(pattern.graph.nodes())
    position = {node: index for index, node in enumerate(pattern_nodes)}
    result = np.zeros((host.number_of_nodes(), host.number_of_nodes()),
                      dtype=np.int64)
    for images in itertools.product(host.nodes(), repeat=len(pattern_nodes)):
        if all(host.has_edge(images[position[a]], images[position[b]])
               for a, b in pattern.graph.edges()):
            result[images[position[root]], images[position[second_root]]] += 1
    return result


class TestTwoRootHomCounts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(BASIS_DIR, 'cycle4_basis.json'), 'rb') as handle:
            basis_info = dill.load(handle)
        spasm = basis_info['SpasmSpace']
        basis = basis_info['basis']
        # The 4-cycle itself is the unique four-vertex basis pattern.
        gid = next(gid for gid in basis if len(spasm[gid].V) == 4)
        cls.pattern = spasm[gid]

    def test_matches_brute_force_and_node_marginal(self):
        host = nx.Graph()
        host.add_nodes_from(range(5))
        host.add_edges_from([
            (0, 1), (1, 2), (2, 3), (3, 0),
            (0, 2), (2, 4),
        ])
        directed_edges = [
            (a, b) for a, b in host.edges()
            for a, b in ((a, b), (b, a))
        ]
        host_df = pd.DataFrame(directed_edges, columns=['s', 't'])

        root, second_root, plan = build_two_root_plan(self.pattern)
        actual = execute_two_root_plan(
            plan, host_df, host.number_of_nodes(), root, second_root
        )
        expected = brute_force_two_root(
            self.pattern, host, root, second_root
        )

        np.testing.assert_array_equal(actual, expected)
        # Adjacent pattern roots imply support only on host edges.
        for u, v in zip(*np.nonzero(actual)):
            self.assertTrue(host.has_edge(int(u), int(v)))
        # Summing out the second root gives the released one-root query.
        state, empty = naive_pandas_plan_exec(
            self.pattern.plan, host_df, sliced_eval={}
        )
        self.assertFalse(empty)
        released = np.zeros(host.number_of_nodes(), dtype=np.int64)
        grouped = state['node$0'].groupby(root)['count'].sum()
        released[grouped.index.to_numpy(dtype=np.int64)] = np.fromiter(
            (int(value) for value in grouped),
            dtype=np.int64,
            count=len(grouped),
        )
        np.testing.assert_array_equal(actual.sum(axis=1), released)


if __name__ == '__main__':
    unittest.main()
