#!/usr/bin/env python3
"""Verify the trainable parameter count of one GraphGPS configuration."""

import argparse
from types import SimpleNamespace

import graphgps  # noqa: F401; register project modules
from torch_geometric.graphgym.config import cfg, load_cfg, set_cfg
from torch_geometric.graphgym.model_builder import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--expected', required=True, type=int)
    args = parser.parse_args()

    set_cfg(cfg)
    cfg.set_new_allowed(True)
    load_cfg(
        cfg,
        SimpleNamespace(
            cfg_file=args.cfg,
            opts=['accelerator', 'cpu'],
        ),
    )
    model = create_model()
    actual = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if actual != args.expected:
        raise ValueError(
            f'Expected {args.expected} trainable parameters, got {actual}'
        )
    print(actual)


if __name__ == '__main__':
    main()
