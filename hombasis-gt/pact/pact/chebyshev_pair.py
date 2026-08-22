"""Chebyshev filtering utilities for edge-supported two-root signals."""

import torch


def total_degree_pairs(max_degree):
    """Return (left_degree, right_degree) in graded lexicographic order."""
    return tuple(
        (left_degree, total_degree - left_degree)
        for total_degree in range(max_degree + 1)
        for left_degree in range(total_degree + 1)
    )


def shifted_normalized_laplacian(edge_index, num_nodes, dtype=torch.float64):
    """Build L_norm - I = -D^{-1/2} A D^{-1/2} for an unweighted graph."""
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=dtype)
    adjacency[edge_index[0], edge_index[1]] = 1
    degree = adjacency.sum(dim=1)
    inverse_sqrt_degree = torch.where(
        degree > 0,
        degree.rsqrt(),
        torch.zeros_like(degree),
    )
    return -(
        inverse_sqrt_degree[:, None]
        * adjacency
        * inverse_sqrt_degree[None, :]
    )


def chebyshev_matrices(operator, max_degree):
    """Evaluate T_0(operator), ..., T_max_degree(operator) by recurrence."""
    matrices = [
        torch.eye(operator.shape[0], dtype=operator.dtype, device=operator.device)
    ]
    if max_degree == 0:
        return tuple(matrices)

    matrices.append(operator)
    for _ in range(1, max_degree):
        matrices.append(2 * operator @ matrices[-1] - matrices[-2])
    return tuple(matrices)


def edge_restricted_pair_responses(
    polynomial_matrices,
    edge_index,
    edge_signals,
    degree_pairs,
):
    """Compute Res_E[P_a Phi_t P_b] without truncating intermediate pairs.

    ``edge_signals[p, t]`` stores Phi_t(x_p, y_p). The input signals must be
    zero away from the listed directed edges. The returned tensor has shape
    ``[num_edges, num_signals, num_degree_pairs]``.
    """
    sources, targets = edge_index
    responses = torch.empty(
        (edge_signals.shape[0], edge_signals.shape[1], len(degree_pairs)),
        dtype=edge_signals.dtype,
        device=edge_signals.device,
    )

    left_factors = {}
    right_factors = {}
    for left_degree, right_degree in degree_pairs:
        if left_degree not in left_factors:
            left_factors[left_degree] = polynomial_matrices[left_degree][
                sources[:, None], sources[None, :]
            ]
        if right_degree not in right_factors:
            right_factors[right_degree] = polynomial_matrices[right_degree][
                targets[None, :], targets[:, None]
            ]

    for response_index, (left_degree, right_degree) in enumerate(degree_pairs):
        edge_kernel = (
            left_factors[left_degree] * right_factors[right_degree]
        )
        responses[:, :, response_index] = edge_kernel @ edge_signals
    return responses
