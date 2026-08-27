#!/usr/bin/env python3
"""Precompute degree-truncated Chebyshev responses for ZINC edge SpecMoSE.

For every graph and two-root input signal Phi_t, this script computes

    Res_E[T_a(L_norm - I) Phi_t T_b(L_norm - I)],  a + b <= K,

where E is the original directed molecular edge set. The all-zero K4 hom-count
template at channel 13 is replaced by the equality relation Delta = I. Pair
responses are never restricted before the final Res_E operation. The output is
a float32 NumPy memmap with shape [edges, templates, polynomial basis].
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from pact.chebyshev_pair import (
    chebyshev_matrices,
    edge_restricted_equality_responses,
    edge_restricted_pair_responses,
    shifted_normalized_laplacian,
    total_degree_pairs,
)


EQUALITY_CHANNEL = 13
EQUALITY_FEATURE_NAME = 'equality_delta'


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(repo_root):
    try:
        return subprocess.check_output(
            ['git', '-C', str(repo_root), 'rev-parse', 'HEAD'],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_args():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / 'hombasis-bench' / 'data' / 'zinc-data'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input',
        type=Path,
        default=data_dir / 'zinc12k_two_root_mose_edge_v1.pt',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=data_dir / 'zinc12k_two_root_mose_eq_cheby_k10_edge_v1',
    )
    parser.add_argument('--max-degree', type=int, default=10)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--progress-every', type=int, default=500)
    parser.add_argument('--threads', type=int, default=1)
    return parser.parse_args()


def _dense_reference(polynomials, edge_index, edge_signals, pair):
    num_nodes = polynomials[0].shape[0]
    sources, targets = edge_index
    dense_signals = torch.zeros(
        (edge_signals.shape[1], num_nodes, num_nodes),
        dtype=edge_signals.dtype,
    )
    dense_signals[:, sources, targets] = edge_signals.T
    left_degree, right_degree = pair
    dense_response = torch.einsum(
        'ij,tjk,kl->til',
        polynomials[left_degree],
        dense_signals,
        polynomials[right_degree],
    )
    return dense_response[:, sources, targets].T


def main():
    args = _parse_args()
    args.input = args.input.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.max_degree < 0:
        raise ValueError('--max-degree must be nonnegative')
    if args.threads < 1:
        raise ValueError('--threads must be positive')
    torch.set_num_threads(args.threads)
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.output_dir.exists():
        raise FileExistsError(f'Output already exists: {args.output_dir}')

    temporary_dir = args.output_dir.with_name(args.output_dir.name + '.tmp')
    if temporary_dir.exists():
        raise FileExistsError(
            f'Incomplete temporary output already exists: {temporary_dir}'
        )

    print(f'Loading trusted local source artifact: {args.input}', flush=True)
    source = torch.load(args.input, weights_only=False, map_location='cpu')
    source_graph_count = int(source['graph_count'])
    graph_count = source_graph_count
    if args.limit is not None:
        graph_count = min(graph_count, args.limit)

    edge_ptr = source['edge_ptr'][:graph_count + 1].to(torch.int64)
    total_edges = int(edge_ptr[-1])
    edge_index = source['edge_index'][:, :total_edges].to(torch.int64)
    node_counts = source['node_counts'][:graph_count].to(torch.int64)
    numerators = source['signal_numerators'][:total_edges].to(torch.int64)
    denominators = source['signal_denominators'].to(torch.int64)
    num_signals = numerators.shape[1]
    if source['feature_names'][EQUALITY_CHANNEL] != 'hom_13':
        raise ValueError('Expected the all-zero K4 template at channel 13')
    if torch.count_nonzero(numerators[:, EQUALITY_CHANNEL]):
        raise ValueError('K4 source channel 13 is not identically zero')
    feature_names = list(source['feature_names'])
    feature_names[EQUALITY_CHANNEL] = EQUALITY_FEATURE_NAME
    degree_pairs = total_degree_pairs(args.max_degree)
    num_basis = len(degree_pairs)

    temporary_dir.mkdir(parents=True)
    response_path = temporary_dir / 'responses.npy'
    responses_memmap = np.lib.format.open_memmap(
        response_path,
        mode='w+',
        dtype=np.float32,
        shape=(total_edges, num_signals, num_basis),
    )

    np.save(temporary_dir / 'edge_ptr.npy', edge_ptr.numpy())
    np.save(temporary_dir / 'edge_index.npy', edge_index.numpy())
    np.save(temporary_dir / 'node_counts.npy', node_counts.numpy())
    np.save(
        temporary_dir / 'basis_pairs.npy',
        np.asarray(degree_pairs, dtype=np.int8),
    )

    print(
        f'Computing {graph_count} graphs, {total_edges} directed edges, '
        f'{num_signals} signals x {num_basis} basis responses...',
        flush=True,
    )
    start = time.perf_counter()
    response_min = float('inf')
    response_max = -float('inf')
    float32_error_sq = 0.0
    reference_sq = 0.0
    float32_max_abs_error = 0.0
    active_template_total = 0
    dense_reference_checks = 0
    reference_graphs = {
        index for index in (0, 1, graph_count // 2, graph_count - 1)
        if 0 <= index < graph_count
    }
    reference_pairs = {
        (0, 0),
        (0, args.max_degree),
        (args.max_degree, 0),
        (args.max_degree // 2, args.max_degree - args.max_degree // 2),
    }

    for graph_index in range(graph_count):
        lower = int(edge_ptr[graph_index])
        upper = int(edge_ptr[graph_index + 1])
        graph_edges = edge_index[:, lower:upper]
        num_nodes = int(node_counts[graph_index])
        edge_signals = (
            numerators[lower:upper].to(torch.float64)
            / denominators.to(torch.float64)
        )

        operator = shifted_normalized_laplacian(
            graph_edges,
            num_nodes,
            dtype=torch.float64,
        )
        if not torch.equal(operator, operator.T):
            raise AssertionError(
                f'Graph {graph_index} is not undirected after COO construction'
            )
        polynomials = chebyshev_matrices(operator, args.max_degree)

        active_templates = edge_signals.ne(0).any(dim=0)
        active_templates[EQUALITY_CHANNEL] = False
        active_template_total += int(active_templates.sum())
        graph_responses = torch.zeros(
            (upper - lower, num_signals, num_basis),
            dtype=torch.float64,
        )
        graph_responses[:, active_templates] = edge_restricted_pair_responses(
            polynomials,
            graph_edges,
            edge_signals[:, active_templates],
            degree_pairs,
        )
        graph_responses[:, EQUALITY_CHANNEL] = (
            edge_restricted_equality_responses(
                polynomials,
                graph_edges,
                degree_pairs,
            )
        )

        torch.testing.assert_close(
            graph_responses[:, :, 0],
            edge_signals,
            rtol=0,
            atol=0,
            msg=lambda message: f'Identity mismatch in graph {graph_index}: {message}',
        )
        if not torch.isfinite(graph_responses).all():
            raise AssertionError(f'Non-finite response in graph {graph_index}')

        if graph_index in reference_graphs:
            for response_index, pair in enumerate(degree_pairs):
                if pair not in reference_pairs:
                    continue
                expected = _dense_reference(
                    polynomials, graph_edges, edge_signals, pair
                )
                equality_expected = (
                    polynomials[pair[0]] @ polynomials[pair[1]]
                )[graph_edges[0], graph_edges[1]]
                expected[:, EQUALITY_CHANNEL] = equality_expected
                torch.testing.assert_close(
                    graph_responses[:, :, response_index],
                    expected,
                    rtol=1e-11,
                    atol=1e-11,
                    msg=lambda message: (
                        f'Dense reference mismatch in graph {graph_index}, '
                        f'pair={pair}: {message}'
                    ),
                )
                dense_reference_checks += 1

        graph_float32 = graph_responses.to(torch.float32)
        quantization_error = graph_float32.to(torch.float64) - graph_responses
        float32_error_sq += float((quantization_error ** 2).sum())
        reference_sq += float((graph_responses ** 2).sum())
        float32_max_abs_error = max(
            float32_max_abs_error,
            float(quantization_error.abs().max()),
        )
        response_min = min(response_min, float(graph_responses.min()))
        response_max = max(response_max, float(graph_responses.max()))
        responses_memmap[lower:upper] = graph_float32.numpy()

        completed = graph_index + 1
        if (
            completed == 1
            or completed % args.progress_every == 0
            or completed == graph_count
        ):
            elapsed = time.perf_counter() - start
            rate = completed / elapsed
            eta = (graph_count - completed) / rate if rate else float('inf')
            print(
                f'[{completed}/{graph_count}] {rate:.2f} graphs/s, '
                f'elapsed={elapsed / 60:.1f} min, ETA={eta / 60:.1f} min',
                flush=True,
            )

    responses_memmap.flush()
    del responses_memmap

    file_hashes = {
        path.name: _sha256(path)
        for path in sorted(temporary_dir.iterdir())
        if path.is_file()
    }
    repo_root = Path(__file__).resolve().parents[2]
    metadata = {
        'format': 'edge_restricted_two_root_mose_equality_chebyshev',
        'format_version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'complete': True,
        'complete_zinc12k': graph_count == 12000,
        'graph_count': graph_count,
        'directed_edge_count': total_edges,
        'input_signal_count': num_signals,
        'input_feature_names': feature_names,
        'replaced_input_channel': {
            'index': EQUALITY_CHANNEL,
            'old_name': source['feature_names'][EQUALITY_CHANNEL],
            'old_relation': 'ordered two-root K4 hom-count (all zero on ZINC)',
            'new_name': EQUALITY_FEATURE_NAME,
            'new_relation': 'Delta(u,v) = 1[u=v]',
        },
        'max_total_degree': args.max_degree,
        'basis_count': num_basis,
        'basis_order': 'graded lexicographic: total degree, then left degree',
        'basis_pairs': [list(pair) for pair in degree_pairs],
        'response_shape': [total_edges, num_signals, num_basis],
        'response_dtype': 'float32',
        'operator': {
            'name': 'shifted normalized Laplacian',
            'normalized_laplacian': 'I - D^{-1/2} A D^{-1/2}',
            'chebyshev_argument': 'L_norm - I = -D^{-1/2} A D^{-1/2}',
            'spectral_interval': [-1, 1],
            'lambda_max_bound': 2,
            'lambda_max_scaling': 'fixed and graph-independent',
            'adjacency': 'unweighted, no added self-loops',
        },
        'response_definition': (
            'T_a(L_norm-I) Phi_t T_b(L_norm-I), a+b<=K; '
            'Phi_13 is equality Delta=I'
        ),
        'storage_semantics': {
            'ordinary_input_pair_support': 'original directed molecular edges',
            'equality_input_pair_support': 'node-pair diagonal',
            'stored_output_support': 'original directed molecular edges',
            'operation': 'final restriction Res_E after full pair filtering',
            'intermediate_edge_truncation': False,
            'full_v_squared_materialized': False,
            'virtual_node_included': False,
        },
        'source': {
            'path': str(args.input),
            'sha256': _sha256(args.input),
            'signal_denominators': denominators.tolist(),
        },
        'files_sha256': file_hashes,
        'generator_source_sha256': {
            'precompute_zinc_two_root_chebyshev.py': _sha256(
                Path(__file__).resolve()
            ),
            'pact/chebyshev_pair.py': _sha256(
                Path(__file__).resolve().parent / 'pact' / 'chebyshev_pair.py'
            ),
        },
        'git_commit': _git_commit(repo_root),
        'environment': {
            'python': sys.version,
            'numpy': np.__version__,
            'torch': str(torch.__version__),
            'torch_num_threads': torch.get_num_threads(),
        },
        'validation': {
            'identity_response_exact_before_float32_cast': True,
            'equality_identity_edge_response_zero': True,
            'equality_dense_reference_via_chebyshev_product': True,
            'finite_responses_all_graphs': True,
            'dense_reference_checks': dense_reference_checks,
            'dense_reference_rtol': 1e-11,
            'dense_reference_atol': 1e-11,
            'float64_response_min': response_min,
            'float64_response_max': response_max,
            'float32_relative_l2_error': (
                (float32_error_sq / reference_sq) ** 0.5
                if reference_sq else 0.0
            ),
            'float32_max_abs_error': float32_max_abs_error,
            'mean_active_templates_per_graph': (
                active_template_total / graph_count
            ),
        },
    }
    with open(temporary_dir / 'metadata.json', 'w') as handle:
        json.dump(metadata, handle, indent=2)
        handle.write('\n')

    response_bytes = response_path.stat().st_size
    os.replace(temporary_dir, args.output_dir)
    elapsed = time.perf_counter() - start
    print(
        f'Saved complete artifact: {args.output_dir}\n'
        f'responses_bytes={response_bytes}, '
        f'wall={elapsed / 60:.2f} min',
        flush=True,
    )


if __name__ == '__main__':
    main()
