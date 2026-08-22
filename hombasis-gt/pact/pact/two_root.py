"""Utilities for exact two-root homomorphism counting with PACT plans.

The cycle-basis artifacts shipped with MoSE contain query plans whose root
tree-decomposition bag is suitable for the released one-root counts.  A
two-root query needs both marked pattern vertices to remain in the final bag.
This module re-roots a copy of the stored decomposition at such a bag and
builds a new, otherwise unchanged, PACT execution plan.
"""

from copy import deepcopy

import numpy as np

from pact.naive_exec import naive_pandas_plan_exec
from pact.planner import node_to_ops


def smallest_adjacent_second_root(pattern, root=0):
    """Return the smallest-labelled pattern neighbour of ``root``."""
    neighbours = list(pattern.graph.neighbors(root))
    if not neighbours:
        raise ValueError(f"Pattern root {root} has no adjacent vertex")
    return min(neighbours)


def _reroot_tree_decomposition(tree, required_vertices):
    """Re-orient a copied tree decomposition at a bag containing the roots."""
    nodes = list(tree.nodes())
    by_id = {id(node): node for node in nodes}
    adjacency = {id(node): [] for node in nodes}

    for node in nodes:
        for child in node.children:
            adjacency[id(node)].append(id(child))
            adjacency[id(child)].append(id(node))

    required_vertices = set(required_vertices)
    candidates = [node for node in nodes
                  if required_vertices.issubset(node.bag)]
    if not candidates:
        raise ValueError(
            f"No decomposition bag contains roots {sorted(required_vertices)}"
        )

    # The choice among valid bags affects runtime, not the count.  Prefer the
    # smallest bag and use its sorted labels as a deterministic tie-breaker.
    new_root = min(
        candidates,
        key=lambda node: (len(node.bag), tuple(sorted(node.bag))),
    )

    def orient(node, parent=None):
        children = []
        for neighbour_id in adjacency[id(node)]:
            neighbour = by_id[neighbour_id]
            if neighbour is parent:
                continue
            children.append(neighbour)
        node.children = children
        for child in children:
            orient(child, node)

    orient(new_root)
    return new_root


def build_two_root_plan(pattern, root=0, second_root=None):
    """Build a PACT plan retaining an ordered pair of pattern roots.

    The input ``pattern`` is not mutated.  When ``second_root`` is omitted,
    the smallest-labelled neighbour of ``root`` is used.
    """
    if second_root is None:
        second_root = smallest_adjacent_second_root(pattern, root)
    if root == second_root:
        raise ValueError("This helper expects two distinct pattern vertices")
    if root not in pattern.graph or second_root not in pattern.graph:
        raise ValueError("Both roots must be vertices of the pattern")

    tree = deepcopy(pattern.td)
    tree = _reroot_tree_decomposition(tree, (root, second_root))
    return root, second_root, node_to_ops(tree)


def execute_two_root_plan(plan, host_df, num_nodes, root, second_root):
    """Return the exact ordered two-root hom-count matrix for one host graph."""
    state, empty = naive_pandas_plan_exec(
        plan,
        host_df,
        sliced_eval={},
    )
    result = np.zeros((num_nodes, num_nodes), dtype=np.int64)
    if empty:
        return result

    grouped = (
        state['node$0']
        .groupby([root, second_root], as_index=False)['count']
        .sum()
    )
    rows = grouped[root].to_numpy(dtype=np.int64)
    cols = grouped[second_root].to_numpy(dtype=np.int64)
    counts = np.fromiter(
        (int(value) for value in grouped['count']),
        dtype=np.int64,
        count=len(grouped),
    )
    result[rows, cols] = counts
    return result
