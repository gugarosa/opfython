# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Share the forest operations used by the public model implementations."""

import numpy as np

import opfython.utils.constants as c
import opfython.utils.exception as e
from opfython.core.heap import Heap
from opfython.core.opf import OPF
from opfython.subgraphs.knn import KNNSubgraph


def _predict_knn(model: OPF, features: np.ndarray, indexes: np.ndarray | None) -> KNNSubgraph:
    subgraph = model.subgraph
    if not isinstance(subgraph, KNNSubgraph):
        raise e.BuildError("`subgraph` should be a KNNSubgraph before prediction.")

    predictions = KNNSubgraph(features, I=indexes)
    model._validate_pre_distances(predictions, subgraph)
    best_k = subgraph.best_k

    distances = np.zeros(best_k + 1)
    neighbours_idx = np.zeros(best_k + 1)

    for node in predictions.nodes:
        cost = -c.FLOAT_MAX
        distances.fill(c.FLOAT_MAX)

        for j, neighbour in enumerate(subgraph.nodes):
            if model.pre_computed_distance:
                distances[best_k] = model.pre_distances[node.idx][neighbour.idx]
            else:
                distances[best_k] = model.distance_fn(node.features, neighbour.features)

            neighbours_idx[best_k] = j
            cur_k = best_k

            # Strict comparisons preserve training-node order for equal distances
            while cur_k > 0 and distances[cur_k] < distances[cur_k - 1]:
                distances[cur_k], distances[cur_k - 1] = distances[cur_k - 1], distances[cur_k]
                neighbours_idx[cur_k], neighbours_idx[cur_k - 1] = neighbours_idx[cur_k - 1], neighbours_idx[cur_k]
                cur_k -= 1

        density = 0.0
        for k in range(best_k):
            density += np.exp(-distances[k] / subgraph.constant)
        density /= best_k
        density = (
            (c.MAX_DENSITY - 1)
            * (density - subgraph.min_density)
            / (subgraph.max_density - subgraph.min_density + c.EPSILON)
        ) + 1

        for k in range(best_k):
            if distances[k] != c.FLOAT_MAX:
                neighbour = subgraph.nodes[int(neighbours_idx[k])]
                current_cost = np.minimum(neighbour.cost, density)
                if current_cost > cost:
                    cost = current_cost
                    node.predicted_label = neighbour.predicted_label
                    node.cluster_label = neighbour.cluster_label

    return predictions


def _grow_minimax_forest(model: OPF, update_labels: bool = False) -> None:
    subgraph = model.subgraph
    if subgraph is None:
        raise e.BuildError("`subgraph` is None.")

    heap = Heap(size=subgraph.n_nodes)

    for i, node in enumerate(subgraph.nodes):
        if node.status == c.PROTOTYPE:
            node.pred = c.NIL
            node.predicted_label = node.label
            heap.cost[i] = 0
            heap.insert(i)
        else:
            heap.cost[i] = c.FLOAT_MAX

    while not heap.is_empty():
        p = heap.remove()
        node = subgraph.nodes[p]
        subgraph.idx_nodes.append(p)
        node.cost = heap.cost[p]

        for q, neighbour in enumerate(subgraph.nodes):
            if p != q and heap.cost[p] < heap.cost[q]:
                if model.pre_computed_distance:
                    weight = model.pre_distances[node.idx][neighbour.idx]
                else:
                    weight = model.distance_fn(node.features, neighbour.features)

                current_cost = np.maximum(heap.cost[p], weight)
                if current_cost < heap.cost[q]:
                    neighbour.pred = p
                    neighbour.predicted_label = node.predicted_label
                    if update_labels:
                        neighbour.label = neighbour.predicted_label

                    heap.update(q, current_cost)
