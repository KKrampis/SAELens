import json
import math

import pytest
import torch

from sae_lens.synthetic.feature_dictionary import (
    FeatureDictionary,
    semantic_initializer,
)
from sae_lens.synthetic.hierarchy.hierarchy import (
    Hierarchy,
    concept_node_to_hierarchy_node,
)
from sae_lens.synthetic.hierarchy.node import HierarchyNode
from sae_lens.synthetic.semantic_labels import (
    ConceptNode,
    load_semantic_dictionary,
)


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# load_semantic_dictionary
# ---------------------------------------------------------------------------


def test_load_semantic_dictionary_parses_simple_tree(tmp_path):
    _write_json(
        tmp_path / "dict.json",
        {
            "trees": [
                {
                    "label": "Root",
                    "alpha": 0.0,
                    "beta": 1.0,
                    "mutually_exclusive_children": True,
                    "children": [
                        {
                            "label": "Child A",
                            "alpha": 0.7,
                            "beta": 0.714,
                            "children": [],
                        },
                        {
                            "label": "Child B",
                            "alpha": 0.4,
                            "beta": 0.917,
                            "children": [],
                        },
                    ],
                }
            ]
        },
    )

    roots = load_semantic_dictionary(str(tmp_path / "dict.json"))

    assert len(roots) == 1
    root = roots[0]
    assert root.label == "Root"
    assert root.alpha == pytest.approx(0.0)
    assert root.beta == pytest.approx(1.0)
    assert root.mutually_exclusive_children is True
    assert len(root.children) == 2
    assert root.children[0].label == "Child A"
    assert root.children[0].alpha == pytest.approx(0.7)
    assert root.children[1].label == "Child B"


def test_load_semantic_dictionary_multiple_trees(tmp_path):
    _write_json(
        tmp_path / "dict.json",
        {
            "trees": [
                {"label": "Tree1", "alpha": 0.0, "beta": 1.0, "children": []},
                {"label": "Tree2", "alpha": 0.0, "beta": 1.0, "children": []},
                {"label": "Tree3", "alpha": 0.0, "beta": 1.0, "children": []},
            ]
        },
    )

    roots = load_semantic_dictionary(str(tmp_path / "dict.json"))
    assert len(roots) == 3
    assert [r.label for r in roots] == ["Tree1", "Tree2", "Tree3"]


def test_load_semantic_dictionary_missing_required_field_raises(tmp_path):
    _write_json(
        tmp_path / "dict.json",
        {"trees": [{"label": "Root", "alpha": 0.5}]},  # missing beta
    )
    with pytest.raises(ValueError, match="Missing required field 'beta'"):
        load_semantic_dictionary(str(tmp_path / "dict.json"))


def test_load_semantic_dictionary_missing_trees_key_raises(tmp_path):
    _write_json(tmp_path / "dict.json", {"nodes": []})
    with pytest.raises(ValueError, match="'trees'"):
        load_semantic_dictionary(str(tmp_path / "dict.json"))


def test_load_semantic_dictionary_defaults_mutually_exclusive_to_false(tmp_path):
    _write_json(
        tmp_path / "dict.json",
        {"trees": [{"label": "Root", "alpha": 0.0, "beta": 1.0, "children": []}]},
    )
    roots = load_semantic_dictionary(str(tmp_path / "dict.json"))
    assert roots[0].mutually_exclusive_children is False


# ---------------------------------------------------------------------------
# concept_node_to_hierarchy_node
# ---------------------------------------------------------------------------


def test_concept_node_to_hierarchy_node_assigns_sequential_indices():
    # DFS preorder: root=0, childA=1, grandchild=2, childB=3
    tree = ConceptNode(
        label="Root",
        alpha=0.0,
        beta=1.0,
        children=[
            ConceptNode(
                label="Child A",
                alpha=0.6,
                beta=0.8,
                children=[
                    ConceptNode(label="Grandchild", alpha=0.5, beta=0.866, children=[])
                ],
            ),
            ConceptNode(label="Child B", alpha=0.4, beta=0.917, children=[]),
        ],
    )

    root_node, next_idx = concept_node_to_hierarchy_node(tree, feature_index_start=0)

    assert root_node.feature_index == 0
    assert root_node.children[0].feature_index == 1
    assert root_node.children[0].children[0].feature_index == 2
    assert root_node.children[1].feature_index == 3
    assert next_idx == 4


def test_concept_node_to_hierarchy_node_preserves_alpha_beta_label():
    node = ConceptNode(label="Concept", alpha=0.65, beta=0.76, children=[])
    h_node, _ = concept_node_to_hierarchy_node(node, feature_index_start=5)

    assert h_node.alpha == pytest.approx(0.65)
    assert h_node.beta == pytest.approx(0.76)
    assert h_node.label == "Concept"
    assert h_node.feature_index == 5


def test_concept_node_to_hierarchy_node_sets_mutually_exclusive():
    tree = ConceptNode(
        label="Parent",
        alpha=0.0,
        beta=1.0,
        mutually_exclusive_children=True,
        children=[
            ConceptNode(label="C1", alpha=0.5, beta=0.866, children=[]),
            ConceptNode(label="C2", alpha=0.5, beta=0.866, children=[]),
        ],
    )
    h_node, _ = concept_node_to_hierarchy_node(tree, feature_index_start=0)
    assert h_node.mutually_exclusive_children is True


def test_concept_node_to_hierarchy_node_offset_start():
    node = ConceptNode(
        label="A",
        alpha=0.0,
        beta=1.0,
        children=[ConceptNode(label="B", alpha=0.5, beta=0.866, children=[])],
    )
    root, next_idx = concept_node_to_hierarchy_node(node, feature_index_start=10)
    assert root.feature_index == 10
    assert root.children[0].feature_index == 11
    assert next_idx == 12


# ---------------------------------------------------------------------------
# semantic_initializer — geometric correctness
# ---------------------------------------------------------------------------


def test_semantic_initializer_encodes_alpha():
    """Child vector cosine similarity to parent must equal alpha when alpha²+beta²=1."""
    alpha = 0.7
    beta = math.sqrt(1 - alpha**2)

    child = HierarchyNode(feature_index=1, alpha=alpha, beta=beta)
    root = HierarchyNode(feature_index=0, children=[child], alpha=0.0, beta=1.0)
    hierarchy = Hierarchy(roots=[root], modifier=None)

    feature_dict = FeatureDictionary(
        num_features=10,
        hidden_dim=256,
        initializer=semantic_initializer(hierarchy, num_features=10),
        seed=0,
    )

    root_vec = feature_dict.feature_vectors[0]
    child_vec = feature_dict.feature_vectors[1]
    cos_sim = (root_vec @ child_vec).item()

    assert cos_sim == pytest.approx(alpha, abs=1e-5)


def test_semantic_initializer_grandchild_encodes_alpha():
    """Grandchild similarity to its parent (depth-1 node) equals grandchild's alpha."""
    alpha_child = 0.6
    beta_child = math.sqrt(1 - alpha_child**2)
    alpha_gc = 0.5
    beta_gc = math.sqrt(1 - alpha_gc**2)

    grandchild = HierarchyNode(feature_index=2, alpha=alpha_gc, beta=beta_gc)
    child = HierarchyNode(
        feature_index=1, alpha=alpha_child, beta=beta_child, children=[grandchild]
    )
    root = HierarchyNode(feature_index=0, children=[child], alpha=0.0, beta=1.0)
    hierarchy = Hierarchy(roots=[root], modifier=None)

    feature_dict = FeatureDictionary(
        num_features=10,
        hidden_dim=512,
        initializer=semantic_initializer(hierarchy, num_features=10),
        seed=42,
    )

    child_vec = feature_dict.feature_vectors[1]
    gc_vec = feature_dict.feature_vectors[2]
    cos_sim = (child_vec @ gc_vec).item()

    assert cos_sim == pytest.approx(alpha_gc, abs=1e-5)


def test_semantic_initializer_free_features_are_unit_vectors():
    """Features outside the hierarchy should be unit vectors after initialization."""
    child = HierarchyNode(feature_index=1, alpha=0.5, beta=math.sqrt(0.75))
    root = HierarchyNode(feature_index=0, children=[child], alpha=0.0, beta=1.0)
    hierarchy = Hierarchy(roots=[root], modifier=None)

    num_features = 10
    feature_dict = FeatureDictionary(
        num_features=num_features,
        hidden_dim=64,
        initializer=semantic_initializer(hierarchy, num_features=num_features),
        seed=7,
    )

    norms = feature_dict.feature_vectors.norm(dim=1)
    assert torch.allclose(norms, torch.ones(num_features), atol=1e-5)
