import unittest

import torch

from pact.chebyshev_pair import (
    chebyshev_matrices,
    edge_restricted_equality_responses,
    edge_restricted_pair_responses,
    shifted_normalized_laplacian,
    total_degree_pairs,
)


class ChebyshevPairTest(unittest.TestCase):
    def setUp(self):
        # An undirected triangle with one attached edge, stored in both directions.
        self.edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 0, 2, 3],
            [1, 0, 2, 1, 0, 2, 3, 2],
        ])
        self.num_nodes = 4
        self.edge_signals = torch.arange(
            1,
            1 + self.edge_index.shape[1] * 2,
            dtype=torch.float64,
        ).reshape(self.edge_index.shape[1], 2)

    def _responses(self, edge_index, edge_signals):
        operator = shifted_normalized_laplacian(
            edge_index, self.num_nodes, dtype=torch.float64
        )
        polynomials = chebyshev_matrices(operator, max_degree=3)
        degree_pairs = total_degree_pairs(3)
        responses = edge_restricted_pair_responses(
            polynomials,
            edge_index,
            edge_signals,
            degree_pairs,
        )
        return operator, polynomials, degree_pairs, responses

    def test_total_degree_basis(self):
        pairs = total_degree_pairs(10)
        self.assertEqual(len(pairs), 66)
        self.assertEqual(pairs[0], (0, 0))
        self.assertTrue(all(a + b <= 10 for a, b in pairs))
        self.assertEqual(len(set(pairs)), len(pairs))

    def test_edge_restriction_matches_dense_response(self):
        _, polynomials, degree_pairs, responses = self._responses(
            self.edge_index, self.edge_signals
        )
        sources, targets = self.edge_index
        dense_signals = torch.zeros(
            (2, self.num_nodes, self.num_nodes), dtype=torch.float64
        )
        dense_signals[:, sources, targets] = self.edge_signals.T

        for response_index, (left_degree, right_degree) in enumerate(degree_pairs):
            dense_response = torch.einsum(
                'ij,tjk,kl->til',
                polynomials[left_degree],
                dense_signals,
                polynomials[right_degree],
            )
            expected = dense_response[:, sources, targets].T
            torch.testing.assert_close(
                responses[:, :, response_index],
                expected,
                rtol=1e-12,
                atol=1e-12,
            )

        torch.testing.assert_close(
            responses[:, :, 0],
            self.edge_signals,
            rtol=0,
            atol=0,
        )

    def test_permutation_equivariance(self):
        _, _, _, expected = self._responses(
            self.edge_index, self.edge_signals
        )
        permutation = torch.tensor([2, 0, 3, 1])
        permuted_edge_index = permutation[self.edge_index]
        _, _, _, actual = self._responses(
            permuted_edge_index, self.edge_signals
        )
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_equality_responses_match_dense_two_sided_filtering(self):
        _, polynomials, degree_pairs, _ = self._responses(
            self.edge_index, self.edge_signals
        )
        actual = edge_restricted_equality_responses(
            polynomials,
            self.edge_index,
            degree_pairs,
        )
        sources, targets = self.edge_index
        identity = torch.eye(self.num_nodes, dtype=torch.float64)
        expected = torch.stack([
            (
                polynomials[left_degree]
                @ identity
                @ polynomials[right_degree]
            )[sources, targets]
            for left_degree, right_degree in degree_pairs
        ], dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertTrue(actual[:, 0].eq(0).all())

    def test_fixed_shifted_spectrum_is_in_chebyshev_interval(self):
        operator, _, _, _ = self._responses(
            self.edge_index, self.edge_signals
        )
        eigenvalues = torch.linalg.eigvalsh(operator)
        self.assertGreaterEqual(float(eigenvalues.min()), -1 - 1e-12)
        self.assertLessEqual(float(eigenvalues.max()), 1 + 1e-12)


if __name__ == '__main__':
    unittest.main()
