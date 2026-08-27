"""Attach the precomputed SpecMoSE edge tensor to a PyG ZINC dataset."""

import json
import logging
from pathlib import Path

import numpy as np
import torch


def _resolve_cache_dir(cache_dir):
    cache_dir = Path(cache_dir)
    if cache_dir.is_absolute():
        return cache_dir
    repository_root = Path(__file__).resolve().parents[3]
    return repository_root / cache_dir


def attach_specmose_edge_cache(
        dataset, cache_dir, expected_templates, expected_basis,
        expected_max_total_degree):
    """Zero-copy attach a mathematically validated response memmap."""
    cache_dir = _resolve_cache_dir(cache_dir)
    with open(cache_dir / 'metadata.json') as handle:
        metadata = json.load(handle)
    if not metadata['complete'] or not metadata['complete_zinc12k']:
        raise ValueError(f'Incomplete SpecMoSE cache: {cache_dir}')
    if metadata['format'] != (
            'edge_restricted_two_root_mose_equality_chebyshev'):
        raise ValueError(f'Unexpected SpecMoSE cache format: {metadata}')
    replacement = metadata.get('replaced_input_channel')
    if replacement != {
        'index': 13,
        'old_name': 'hom_13',
        'old_relation': 'ordered two-root K4 hom-count (all zero on ZINC)',
        'new_name': 'equality_delta',
        'new_relation': 'Delta(u,v) = 1[u=v]',
    }:
        raise ValueError('SpecMoSE cache has an unexpected equality channel')
    if metadata['input_signal_count'] != expected_templates:
        raise ValueError('SpecMoSE template count does not match config')
    source_basis = metadata['basis_count']
    source_max_total_degree = metadata['max_total_degree']
    if source_basis < expected_basis:
        raise ValueError('SpecMoSE cache has too few basis responses')
    if source_max_total_degree < expected_max_total_degree:
        raise ValueError('SpecMoSE cache has insufficient total degree')

    basis_pairs = np.load(cache_dir / 'basis_pairs.npy')
    expected_pairs = np.asarray(
        [
            (left_degree, total_degree - left_degree)
            for total_degree in range(expected_max_total_degree + 1)
            for left_degree in range(total_degree + 1)
        ],
        dtype=basis_pairs.dtype,
    )
    if basis_pairs.shape != (source_basis, 2):
        raise ValueError(f'Unexpected basis-pair shape: {basis_pairs.shape}')
    if not np.array_equal(basis_pairs[:expected_basis], expected_pairs):
        raise ValueError('Requested SpecMoSE basis is not a graded prefix')
    if metadata['basis_pairs'] != basis_pairs.tolist():
        raise ValueError('SpecMoSE basis metadata disagrees with basis file')

    operator = metadata['operator']
    if operator['chebyshev_argument'] != (
            'L_norm - I = -D^{-1/2} A D^{-1/2}'):
        raise ValueError('SpecMoSE cache uses an unexpected operator')
    storage = metadata['storage_semantics']
    if storage != {
        'ordinary_input_pair_support': 'original directed molecular edges',
        'equality_input_pair_support': 'node-pair diagonal',
        'stored_output_support': 'original directed molecular edges',
        'operation': 'final restriction Res_E after full pair filtering',
        'intermediate_edge_truncation': False,
        'full_v_squared_materialized': False,
        'virtual_node_included': False,
    }:
        raise ValueError('SpecMoSE cache uses unexpected storage semantics')

    edge_ptr = np.load(cache_dir / 'edge_ptr.npy', mmap_mode='r')
    edge_index = np.load(cache_dir / 'edge_index.npy', mmap_mode='r')
    node_counts = np.load(cache_dir / 'node_counts.npy', mmap_mode='r')
    if len(dataset) != metadata['graph_count']:
        raise ValueError(
            f'SpecMoSE cache has {metadata["graph_count"]} graphs, '
            f'but dataset has {len(dataset)}'
        )
    if edge_ptr.shape != (len(dataset) + 1,):
        raise ValueError(f'Unexpected edge pointer shape: {edge_ptr.shape}')
    if node_counts.shape != (len(dataset),):
        raise ValueError(f'Unexpected node-count shape: {node_counts.shape}')
    if edge_index.shape != (2, int(edge_ptr[-1])):
        raise ValueError(f'Unexpected cached edge shape: {edge_index.shape}')
    if int(edge_ptr[0]) != 0 or np.any(edge_ptr[1:] < edge_ptr[:-1]):
        raise ValueError('SpecMoSE edge pointer is not monotone from zero')

    for graph_index in range(len(dataset)):
        data = dataset.get(graph_index)
        lower = int(edge_ptr[graph_index])
        upper = int(edge_ptr[graph_index + 1])
        if data.num_nodes != int(node_counts[graph_index]):
            raise ValueError(f'Node-count mismatch in graph {graph_index}')
        cached_edges = torch.from_numpy(
            np.asarray(edge_index[:, lower:upper]).copy()
        )
        if not torch.equal(data.edge_index, cached_edges):
            raise ValueError(f'Edge-order mismatch in graph {graph_index}')

    responses = np.load(cache_dir / 'responses.npy', mmap_mode='c')
    source_shape = (
        int(edge_ptr[-1]), expected_templates, source_basis
    )
    if tuple(metadata['response_shape']) != source_shape:
        raise ValueError('SpecMoSE response metadata disagrees with config')
    if responses.shape != source_shape or responses.dtype != np.float32:
        raise ValueError(
            f'Unexpected SpecMoSE response tensor: '
            f'shape={responses.shape}, dtype={responses.dtype}'
        )
    if int(edge_ptr[-1]) != responses.shape[0]:
        raise ValueError('SpecMoSE edge pointer does not cover all responses')
    if not torch.equal(
        dataset.slices['edge_attr'], torch.from_numpy(edge_ptr.copy())
    ):
        raise ValueError('Dataset edge slices do not match SpecMoSE edge slices')

    # Keep the memmap alive and force future get() calls to slice dataset.data.
    active_responses = responses[:, :, :expected_basis]
    dataset._specmose_memmap = responses
    dataset._data_list = None
    dataset._data.edge_specmose = torch.from_numpy(active_responses)
    dataset.slices['edge_specmose'] = dataset.slices['edge_attr'].clone()
    logging.info(
        'Attached SpecMoSE edge cache %s with active shape %s '
        '(source shape %s)',
        cache_dir,
        active_responses.shape,
        responses.shape,
    )
    return dataset
