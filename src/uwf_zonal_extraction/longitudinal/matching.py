"""Vessel-graph matching across visits — v2 STUB.

Sketch:
  1. Build a graph per visit: nodes = bifurcation points near OD,
     edges = vessel segments with (length, mean_w, A/V) attributes.
  2. Extract a topology descriptor for each major vessel from the first
     3 bifurcations.
  3. Bipartite match descriptors V1 ↔ Vk with a cost that combines
     topology-distance and spatial-distance (DD units).
  4. Measure matched vessels at both visits, apply Knudtson to the same
     set.

Install `uwf-zonal-extraction[matching]` (networkx) before using.
"""

from __future__ import annotations


def match_vessels_across_visits(*args, **kwargs):
    raise NotImplementedError(
        "Vessel-graph matching is scoped out of v1. Install the [matching] "
        "extra and track the issue for v2."
    )
