#!/usr/bin/env python3
"""Precompute the ordered two-root MoSE signals for ZINC-12K.

For every cycle homomorphism-basis pattern F, the first root is the released
MoSE root 0 and the second root is the smallest-labelled neighbour of 0.  The
two roots are therefore adjacent, so the complete raw pair signal is supported
on the original directed molecular edges and can be stored losslessly in COO
edge order.

The output contains 42 exact rational channels per directed edge:
  * 36 selected Spasm(C7) union Spasm(C8) rooted hom-count channels;
  * 6 C3,...,C8 multi-hom channels formed with the released basis
    coefficients.

All values are stored as int64 numerators plus one denominator per channel.
"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

import dill
import numpy as np
import pandas as pd
import torch

from pact.two_root import build_two_root_plan, execute_two_root_plan


OFFICIAL_SELECTED_INDICES = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    15, 20, 21, 22, 24, 25, 27, 29, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 45, 46,
]
CYCLE_LENGTHS = tuple(range(3, 9))


_CYCLE_QUERIES = None
_HOM_C7 = None
_HOM_C8 = None
_MULTIHOM = None
_SUBGRAPH = None


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _load_cycle_queries(basis_dir):
    queries = {}
    for cycle_length in CYCLE_LENGTHS:
        basis_path = basis_dir / f'cycle{cycle_length}_basis.json'
        with open(basis_path, 'rb') as handle:
            basis_info = dill.load(handle)
        spasm = basis_info['SpasmSpace']
        basis = basis_info['basis']
        ordered_ids = [
            gid for gid in basis
            if spasm[gid].td.ghw >= 1
        ]

        cycle_queries = []
        for basis_index, gid in enumerate(ordered_ids):
            pattern = spasm[gid]
            root, second_root, plan = build_two_root_plan(pattern, root=0)
            coefficient = basis[gid]
            cycle_queries.append({
                'basis_index': basis_index,
                'pattern_id': str(gid),
                'root': root,
                'second_root': second_root,
                'plan': plan,
                'coefficient_numerator': int(coefficient.numerator),
                'coefficient_denominator': int(coefficient.denominator),
                'vertices': [int(v) for v in sorted(pattern.V)],
                'edges': [
                    [int(a), int(b)]
                    for a, b in sorted(
                        tuple(sorted(edge)) for edge in pattern.E
                    )
                ],
            })
        queries[cycle_length] = cycle_queries
    return queries


def _cycle_denominator(queries):
    return math.lcm(*[
        query['coefficient_denominator'] for query in queries
    ])


def _host_dataframe(edge_index):
    return pd.DataFrame({
        's': edge_index[0],
        't': edge_index[1],
    }).drop_duplicates()


def _node_rows_from_edge_values(edge_sources, edge_values, num_nodes):
    rows = np.zeros((num_nodes, edge_values.shape[1]), dtype=np.int64)
    np.add.at(rows, edge_sources, edge_values)
    return rows


def _official_node_hom_matrix(graph_index):
    graph_key = str(graph_index)
    c7 = _HOM_C7[graph_key]['homcounts']
    c8 = _HOM_C8[graph_key]['homcounts']
    num_nodes = len(_HOM_C7[graph_key]['features'])
    return np.asarray([
        c7[str(vertex)] + c8[str(vertex)]
        for vertex in range(num_nodes)
    ], dtype=np.int64)


def _official_multihom_matrix(graph_index, num_nodes):
    graph = _MULTIHOM[str(graph_index)]
    return np.asarray([
        graph[str(vertex)] for vertex in range(num_nodes)
    ], dtype=np.float64)


def _process_graph(graph_index):
    graph_key = str(graph_index)
    host = _HOM_C7[graph_key]
    if host['edge_index'] != _HOM_C8[graph_key]['edge_index']:
        raise AssertionError(f'C7/C8 edge order mismatch in graph {graph_index}')

    edge_index = np.asarray(host['edge_index'], dtype=np.int64)
    edge_sources, edge_targets = edge_index
    num_edges = edge_index.shape[1]
    num_nodes = len(host['features'])

    directed_edges = list(zip(edge_sources.tolist(), edge_targets.tolist()))
    if len(set(directed_edges)) != num_edges:
        raise AssertionError(f'Duplicate directed edges in graph {graph_index}')

    host_df = _host_dataframe(edge_index)
    raw_cycle_edge_counts = {}
    cycle_numerators = np.zeros(
        (num_edges, len(CYCLE_LENGTHS)), dtype=np.int64
    )

    for cycle_column, cycle_length in enumerate(CYCLE_LENGTHS):
        queries = _CYCLE_QUERIES[cycle_length]
        denominator = _cycle_denominator(queries)
        edge_counts_for_cycle = []
        derived_numerator = np.zeros(num_edges, dtype=np.int64)

        for query in queries:
            pair_matrix = execute_two_root_plan(
                query['plan'],
                host_df,
                num_nodes,
                query['root'],
                query['second_root'],
            )
            edge_values = pair_matrix[edge_sources, edge_targets]
            if int(edge_values.sum()) != int(pair_matrix.sum()):
                raise AssertionError(
                    f'Non-edge support for adjacent roots: graph={graph_index}, '
                    f'C{cycle_length}, basis={query["basis_index"]}'
                )
            edge_counts_for_cycle.append(edge_values)

            weight = (
                query['coefficient_numerator']
                * (denominator // query['coefficient_denominator'])
            )
            derived_numerator += weight * edge_values

        raw_cycle_edge_counts[cycle_length] = np.stack(
            edge_counts_for_cycle, axis=1
        )
        cycle_numerators[:, cycle_column] = derived_numerator

    raw_c7_c8 = np.concatenate(
        [raw_cycle_edge_counts[7], raw_cycle_edge_counts[8]],
        axis=1,
    )
    selected_hom = raw_c7_c8[:, OFFICIAL_SELECTED_INDICES]
    signal_numerators = np.concatenate(
        [selected_hom, cycle_numerators], axis=1
    )

    official_hom = _official_node_hom_matrix(graph_index)
    actual_hom_rows = _node_rows_from_edge_values(
        edge_sources, raw_c7_c8, num_nodes
    )
    if not np.array_equal(actual_hom_rows, official_hom):
        mismatch = np.argwhere(actual_hom_rows != official_hom)[0]
        raise AssertionError(
            f'Raw hom row-marginal mismatch: graph={graph_index}, '
            f'node={int(mismatch[0])}, channel={int(mismatch[1])}, '
            f'actual={int(actual_hom_rows[tuple(mismatch)])}, '
            f'expected={int(official_hom[tuple(mismatch)])}'
        )

    cycle_denominators = np.asarray([
        _cycle_denominator(_CYCLE_QUERIES[length])
        for length in CYCLE_LENGTHS
    ], dtype=np.int64)
    actual_cycle_rows = _node_rows_from_edge_values(
        edge_sources, cycle_numerators, num_nodes
    )
    official_multihom = _official_multihom_matrix(graph_index, num_nodes)
    scaled_multihom = official_multihom * cycle_denominators[None, :]
    rounded_multihom = np.rint(scaled_multihom).astype(np.int64)
    if np.max(np.abs(scaled_multihom - rounded_multihom), initial=0.0) > 1e-7:
        raise AssertionError(
            f'Official multihom is not rational at expected denominators: '
            f'graph={graph_index}'
        )
    if not np.array_equal(actual_cycle_rows, rounded_multihom):
        mismatch = np.argwhere(actual_cycle_rows != rounded_multihom)[0]
        raise AssertionError(
            f'Cycle row-marginal mismatch: graph={graph_index}, '
            f'node={int(mismatch[0])}, cycle=C{int(mismatch[1]) + 3}'
        )

    subcounts = np.asarray(
        list(_SUBGRAPH[graph_key]['subcounts'].values()),
        dtype=np.int64,
    )[:, :len(CYCLE_LENGTHS)]
    cycle_lengths = np.asarray(CYCLE_LENGTHS, dtype=np.int64)
    incidence_totals = subcounts.sum(axis=0)
    if np.any(incidence_totals % cycle_lengths != 0):
        raise AssertionError(
            f'Cycle incidence is not divisible by cycle length: graph={graph_index}'
        )
    graph_cycle_counts = incidence_totals // cycle_lengths
    if not np.array_equal(
        cycle_numerators.sum(axis=0),
        graph_cycle_counts * cycle_denominators,
    ):
        raise AssertionError(
            f'Cycle subgraph total mismatch: graph={graph_index}'
        )

    return {
        'graph_index': graph_index,
        'num_nodes': num_nodes,
        'edge_index': edge_index,
        'signal_numerators': signal_numerators,
        'max_hom': int(selected_hom.max(initial=0)),
        'max_abs_cycle_numerator': int(
            np.abs(cycle_numerators).max(initial=0)
        ),
    }


def _pattern_metadata(queries, source_offset=None, selected_indices=None):
    records = []
    for query in queries:
        record = {
            key: value for key, value in query.items()
            if key != 'plan'
        }
        if source_offset is not None:
            record['concatenated_source_index'] = (
                source_offset + query['basis_index']
            )
        records.append(record)
    if selected_indices is None:
        return records
    return [records[index] for index in selected_indices]


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
    default_data = script_dir.parent / 'hombasis-bench' / 'data' / 'zinc-data'
    default_basis = script_dir / 'bases' / 'cycles'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=default_data)
    parser.add_argument('--basis-dir', type=Path, default=default_basis)
    parser.add_argument(
        '--output', type=Path,
        default=default_data / 'zinc12k_two_root_mose_edge_v1.pt',
    )
    parser.add_argument('--workers', type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument('--chunksize', type=int, default=4)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    global _CYCLE_QUERIES, _HOM_C7, _HOM_C8, _MULTIHOM, _SUBGRAPH

    args = _parse_args()
    args.data_dir = args.data_dir.resolve()
    args.basis_dir = args.basis_dir.resolve()
    args.output = args.output.resolve()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f'{args.output} already exists; pass --overwrite to replace it'
        )
    if args.workers < 1:
        raise ValueError('--workers must be positive')

    source_paths = {
        'hom_c7': args.data_dir / 'zinc_with_homs_c7.json',
        'hom_c8': args.data_dir / 'zinc_with_homs_c8.json',
        'multihom_c3_c8': args.data_dir / 'zinc_3to8C_multhom.json',
        'subgraph_c3_c10': args.data_dir / 'zinc_3to10C_subgraph.json',
    }
    for path in [*source_paths.values(), *[
        args.basis_dir / f'cycle{length}_basis.json'
        for length in CYCLE_LENGTHS
    ]]:
        if not path.is_file():
            raise FileNotFoundError(path)

    print('Loading official ZINC and cycle-basis artifacts...', flush=True)
    _CYCLE_QUERIES = _load_cycle_queries(args.basis_dir)
    with open(source_paths['hom_c7']) as handle:
        _HOM_C7 = json.load(handle)
    with open(source_paths['hom_c8']) as handle:
        _HOM_C8 = json.load(handle)
    with open(source_paths['multihom_c3_c8']) as handle:
        _MULTIHOM = json.load(handle)
    with open(source_paths['subgraph_c3_c10']) as handle:
        _SUBGRAPH = json.load(handle)

    graph_count = len(_HOM_C7)
    if not (len(_HOM_C8) == len(_MULTIHOM) == len(_SUBGRAPH) == graph_count):
        raise AssertionError('The four official ZINC artifacts differ in graph count')
    if args.limit is not None:
        graph_count = min(graph_count, args.limit)

    cycle_denominators = np.asarray([
        _cycle_denominator(_CYCLE_QUERIES[length])
        for length in CYCLE_LENGTHS
    ], dtype=np.int64)
    signal_denominators = np.concatenate([
        np.ones(len(OFFICIAL_SELECTED_INDICES), dtype=np.int64),
        cycle_denominators,
    ])

    print(
        f'Computing {graph_count} graphs with {args.workers} workers; '
        f'{sum(len(v) for v in _CYCLE_QUERIES.values())} two-root queries/graph...',
        flush=True,
    )
    start = time.perf_counter()
    results = []
    maxima = {'hom': 0, 'cycle_numerator': 0}
    context = mp.get_context('fork')
    with context.Pool(args.workers) as pool:
        iterator = pool.imap(
            _process_graph,
            range(graph_count),
            chunksize=args.chunksize,
        )
        for completed, result in enumerate(iterator, start=1):
            results.append(result)
            maxima['hom'] = max(maxima['hom'], result['max_hom'])
            maxima['cycle_numerator'] = max(
                maxima['cycle_numerator'],
                result['max_abs_cycle_numerator'],
            )
            if completed == 1 or completed % 100 == 0 or completed == graph_count:
                elapsed = time.perf_counter() - start
                rate = completed / elapsed
                eta = (graph_count - completed) / rate if rate else float('inf')
                print(
                    f'[{completed}/{graph_count}] {rate:.2f} graphs/s, '
                    f'elapsed={elapsed / 60:.1f} min, ETA={eta / 60:.1f} min',
                    flush=True,
                )

    edge_ptr = [0]
    node_counts = []
    edge_indices = []
    signal_numerators = []
    for expected_index, result in enumerate(results):
        if result['graph_index'] != expected_index:
            raise AssertionError('Multiprocessing results are out of graph order')
        node_counts.append(result['num_nodes'])
        edge_indices.append(torch.from_numpy(result['edge_index']))
        signal_numerators.append(torch.from_numpy(result['signal_numerators']))
        edge_ptr.append(edge_ptr[-1] + result['edge_index'].shape[1])

    c7_count = len(_CYCLE_QUERIES[7])
    selected_pattern_metadata = []
    c7_c8_metadata = (
        _pattern_metadata(_CYCLE_QUERIES[7], source_offset=0)
        + _pattern_metadata(_CYCLE_QUERIES[8], source_offset=c7_count)
    )
    for selected_index in OFFICIAL_SELECTED_INDICES:
        selected_pattern_metadata.append(c7_c8_metadata[selected_index])

    repo_root = Path(__file__).resolve().parents[2]
    payload = {
        'format': 'mose_two_root_ordered_edge_signals',
        'format_version': 1,
        'graph_count': graph_count,
        'complete_zinc12k': graph_count == 12000,
        'node_counts': torch.tensor(node_counts, dtype=torch.int16),
        'edge_ptr': torch.tensor(edge_ptr, dtype=torch.int64),
        'edge_index': torch.cat(edge_indices, dim=1).to(torch.int64),
        'signal_numerators': torch.cat(signal_numerators, dim=0).to(torch.int64),
        'signal_denominators': torch.from_numpy(signal_denominators),
        'feature_names': (
            [f'hom_{index:02d}' for index in range(36)]
            + [f'cycle_derived_C{length}' for length in CYCLE_LENGTHS]
        ),
        'metadata': {
            'first_root': 0,
            'second_root_rule': 'minimum-labelled neighbour of root 0',
            'ordered_roots': True,
            'roots_are_adjacent': True,
            'host_pair_storage': 'original directed edge_index order',
            'virtual_node_included': False,
            'official_selected_indices': OFFICIAL_SELECTED_INDICES,
            'selected_hom_patterns': selected_pattern_metadata,
            'cycle_basis_patterns': {
                str(length): _pattern_metadata(_CYCLE_QUERIES[length])
                for length in CYCLE_LENGTHS
            },
            'cycle_denominators': cycle_denominators.tolist(),
            'source_sha256': {
                name: _sha256(path) for name, path in source_paths.items()
            },
            'basis_sha256': {
                f'C{length}': _sha256(
                    args.basis_dir / f'cycle{length}_basis.json'
                )
                for length in CYCLE_LENGTHS
            },
            'git_commit': _git_commit(repo_root),
            'generator_source_sha256': {
                'precompute_zinc_two_root.py': _sha256(Path(__file__).resolve()),
                'pact/two_root.py': _sha256(
                    Path(__file__).resolve().parent / 'pact' / 'two_root.py'
                ),
            },
            'environment': {
                'python': sys.version,
                'numpy': np.__version__,
                'pandas': pd.__version__,
                'torch': torch.__version__,
                'dill': dill.__version__,
            },
            'validation': {
                'raw_hom_row_marginals': 'exact for all C7/C8 basis channels',
                'cycle_row_marginals': 'exact rational match to released C3-C8 multihom',
                'cycle_graph_totals': 'exact match to released C3-C8 subgraph counts',
                'non_edge_mass': 0,
                'max_hom_count': maxima['hom'],
                'max_abs_cycle_numerator': maxima['cycle_numerator'],
            },
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + '.tmp')
    torch.save(payload, temporary_output)
    os.replace(temporary_output, args.output)
    digest = _sha256(args.output)
    elapsed = time.perf_counter() - start
    print(
        f'Saved {args.output}\n'
        f'bytes={args.output.stat().st_size}, sha256={digest}\n'
        f'graphs={graph_count}, directed_edges={edge_ptr[-1]}, '
        f'wall={elapsed / 60:.2f} min',
        flush=True,
    )


if __name__ == '__main__':
    main()
