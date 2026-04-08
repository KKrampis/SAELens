"""
Semantic dictionary loader for hierarchical feature geometry.

Parses a JSON file defining a forest of concept trees, where each node specifies
a label and α/β mixing coefficients encoding parent-child geometric similarity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConceptNode:
    """
    A node in a user-defined semantic concept tree.

    Attributes:
        label: Human-readable concept name.
        alpha: Cosine similarity to parent direction. Use 0.0 for root nodes.
        beta: Orthogonal mixing weight. Typically sqrt(1 - alpha²) but can be
            set independently for ablations.
        mutually_exclusive_children: If True, at most one child fires per sample.
        children: Child concept nodes.
    """

    label: str
    alpha: float
    beta: float
    mutually_exclusive_children: bool = False
    children: list[ConceptNode] = field(default_factory=list)


def load_semantic_dictionary(path: str) -> list[ConceptNode]:
    """
    Load root ConceptNodes from a JSON file.

    The JSON must have a top-level "trees" array. Each element is a node with
    fields: label (str), alpha (float), beta (float),
    mutually_exclusive_children (bool, optional), children (array, optional).

    Args:
        path: Path to the JSON file.

    Returns:
        List of root ConceptNodes, one per tree.

    Raises:
        ValueError: If the JSON is missing required fields or has an invalid structure.
    """
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, dict) or "trees" not in data:
        raise ValueError("JSON must have a top-level 'trees' array")

    trees = data["trees"]
    if not isinstance(trees, list):
        raise ValueError("'trees' must be an array")

    return [_parse_concept_node(tree) for tree in trees]


def _parse_concept_node(d: Any) -> ConceptNode:
    if not isinstance(d, dict):
        raise ValueError(f"Each node must be a JSON object, got: {type(d)}")

    for key in ("label", "alpha", "beta"):
        if key not in d:
            raise ValueError(f"Missing required field '{key}' in node: {d}")

    return ConceptNode(
        label=str(d["label"]),
        alpha=float(d["alpha"]),
        beta=float(d["beta"]),
        mutually_exclusive_children=bool(d.get("mutually_exclusive_children", False)),
        children=[_parse_concept_node(c) for c in d.get("children", [])],
    )
