# Semantic Dictionaries

A semantic dictionary encodes parent-child concept relationships directly into the geometry of feature vectors. Instead of initializing feature vectors to be as orthogonal as possible, each child feature direction is constructed as a blend of its parent's direction and a new orthogonal component:

```
d_child = α · d_parent + β · d_⊥
```

- **α** (semantic similarity coefficient): cosine similarity between child and parent directions after normalization. Higher values mean the child is geometrically closer to its parent.
- **β** (orthogonal mixing weight): controls how much of the child's direction is unique. Typically `β = sqrt(1 - α²)`, which ensures `cos(θ_child, parent) = α` exactly.
- **d_⊥**: a unit vector orthogonal to the parent direction, derived via Gram-Schmidt.

Root nodes keep random unit vectors. Activation constraints from the existing hierarchy system still apply: children only fire when their parent is active (`c_child ← c_child · 1[c_parent > 0]`), and siblings can be mutually exclusive.

Features not defined in the JSON are **free/non-hierarchical**: they fire independently and their vectors are orthogonalized using `orthogonalize_embeddings()`, exactly as in the standard SynthSAEBench approach.

<!-- prettier-ignore-start -->
!!! info "Beta feature"
    Semantic dictionary support should be considered in beta. The API may change over the next few months. If this is a concern, pin your SAELens version.
<!-- prettier-ignore-end -->

---

## Defining a Semantic Dictionary via JSON

The full hierarchy — labels, α, β, and tree topology — is supplied as a JSON file. This keeps the concept taxonomy outside the code, so taxonomies can be swapped without modifying SAELens.

The `"trees"` array contains however many hierarchical trees you want to define. The remaining `num_features − N_hierarchical` feature slots are automatically free features. You do not need to enumerate them.

**Example (`semantic_dictionary.json`):**

```json
{
  "trees": [
    {
      "label": "Deceptive Reasoning",
      "alpha": 0.0,
      "beta": 1.0,
      "mutually_exclusive_children": true,
      "children": [
        {
          "label": "Goal Misrepresentation",
          "alpha": 0.7,
          "beta": 0.714,
          "mutually_exclusive_children": false,
          "children": [
            { "label": "Framing Manipulation", "alpha": 0.6, "beta": 0.8,   "children": [] },
            { "label": "Selective Omission",   "alpha": 0.5, "beta": 0.866, "children": [] }
          ]
        },
        {
          "label": "Reward Hacking",
          "alpha": 0.4,
          "beta": 0.917,
          "mutually_exclusive_children": false,
          "children": [
            { "label": "Proxy Gaming",               "alpha": 0.6, "beta": 0.8,   "children": [] },
            { "label": "Specification Exploitation", "alpha": 0.5, "beta": 0.866, "children": [] }
          ]
        }
      ]
    }
  ]
}
```

**JSON node fields:**

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | Human-readable concept name |
| `alpha` | float | Cosine similarity to parent direction. Use `0.0` for root nodes. |
| `beta` | float | Orthogonal mixing weight. Typically `sqrt(1-α²)` but can be set independently for ablations. |
| `mutually_exclusive_children` | bool | If `true`, at most one child fires per sample. Optional, defaults to `false`. |
| `children` | array | Nested child nodes. Empty list `[]` for leaf nodes. |

`alpha` and `beta` need not satisfy `α² + β² = 1` strictly — the initializer L2-normalizes the result. However, when `α² + β² = 1`, the cosine similarity between child and parent exactly equals `α`.

---

## Configuration

Pass the JSON path via `HierarchyConfig.semantic_dictionary_path`. When set, `generate_hierarchy()` reads the JSON instead of generating a random forest, and the `total_root_nodes`, `branching_factor`, and `max_depth` fields of `HierarchyConfig` are ignored.

```python
from sae_lens.synthetic import (
    SyntheticModel,
    SyntheticModelConfig,
    HierarchyConfig,
    ZipfianFiringProbabilityConfig,
    LinearMagnitudeConfig,
    FoldedNormalMagnitudeConfig,
)

cfg = SyntheticModelConfig(
    num_features=16_384,
    hidden_dim=768,

    firing_probability=ZipfianFiringProbabilityConfig(
        exponent=0.5,
        max_prob=0.4,
        min_prob=5e-4,
    ),

    hierarchy=HierarchyConfig(
        # Topology comes from the JSON — branching/depth fields are ignored
        semantic_dictionary_path="semantic_dictionary.json",
        # These still apply to all nodes read from JSON:
        compensate_probabilities=True,
        scale_children_by_parent=True,
    ),

    # semantic_geometry is set automatically when semantic_dictionary_path is provided.
    # Set explicitly to True if you want semantic geometry with a manually built hierarchy.
    semantic_geometry=True,

    mean_firing_magnitudes=LinearMagnitudeConfig(start=5.0, end=4.0),
    std_firing_magnitudes=FoldedNormalMagnitudeConfig(mean=0.5, std=0.5),

    seed=42,
)

model = SyntheticModel(cfg)
hidden_activations = model.sample(batch_size=1024)
```

**New fields added to existing configs:**

| Config | Field | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `HierarchyConfig` | `semantic_dictionary_path` | `str \| None` | `None` | Path to JSON file. Overrides random forest generation when set. |
| `SyntheticModelConfig` | `semantic_geometry` | `bool` | `False` | Use `semantic_initializer()` instead of `orthogonal_initializer()`. Auto-enabled when `semantic_dictionary_path` is set. |

---

## Feature Index Assignment and Free Features

Feature indices are assigned depth-first (root first, then children left to right) for each tree in the `"trees"` array, processed in order. For example, with `num_features=16384` and a JSON file containing 50 trees of depth 3, branching factor 4 (50 × 85 = 4,250 hierarchical nodes):

```
Features 0 – 4,249      → hierarchical, geometric α/β initialization
Features 4,250 – 16,383 → free, orthogonalized initialization
```

A complete 4-ary tree of depth 3 has 1 + 4 + 16 + 64 = **85 nodes** per tree. 128 such trees cover 10,880 hierarchical features, matching the [SyntheticLLMs benchmark](https://kkrampis.github.io/SyntheticLLMs/) setup.

`generate_hierarchy()` raises a `ValueError` if the JSON contains more nodes than `num_features`.

---

## How Feature Vectors Are Built

When `semantic_geometry=True` (or auto-detected), `semantic_initializer()` constructs feature vectors as follows:

1. **Root nodes** retain their initial random unit vectors (L2-normalized).
2. **Child nodes** are processed breadth-first. For a node with parent direction `d_parent` and fields `alpha=α`, `beta=β`:
    - Sample `d_⊥` orthogonal to `d_parent` via Gram-Schmidt applied to the node's random initial vector.
    - Compute `d_child = α · d_parent + β · d_⊥`.
    - L2-normalize `d_child`.
3. All vectors are assembled into a `(num_features, hidden_dim)` matrix.
4. Free features (indices beyond the hierarchy) are orthogonalized using `orthogonalize_embeddings()`.

When `α² + β² = 1`, the cosine similarity between child and parent is exactly `α` after normalization.

---

## Loading the JSON in Code

You can also load and inspect the JSON directly via `load_semantic_dictionary()`:

```python
from sae_lens.synthetic import load_semantic_dictionary

roots = load_semantic_dictionary("semantic_dictionary.json")

for root in roots:
    print(root.label, "→", [c.label for c in root.children])
```

`ConceptNode` mirrors the JSON schema:

```python
@dataclass
class ConceptNode:
    label: str
    alpha: float
    beta: float
    mutually_exclusive_children: bool
    children: list["ConceptNode"]
```

`concept_node_to_hierarchy_node()` converts a `ConceptNode` tree to a `HierarchyNode` tree with sequential feature indices, which you can use to build hierarchies programmatically:

```python
from sae_lens.synthetic import ConceptNode, concept_node_to_hierarchy_node

root_concept = ConceptNode(
    label="Sycophancy",
    alpha=0.0,
    beta=1.0,
    mutually_exclusive_children=True,
    children=[
        ConceptNode(label="Approval Seeking",  alpha=0.65, beta=0.76,  children=[]),
        ConceptNode(label="Opinion Mirroring", alpha=0.55, beta=0.835, children=[]),
    ],
)

hierarchy_node, next_index = concept_node_to_hierarchy_node(
    root_concept, feature_index_start=0
)
```

---

## Data Flow

```
semantic_dictionary.json   (N trees, any number up to num_features nodes total)
        │
        ▼
load_semantic_dictionary()                # sae_lens.synthetic.semantic_labels
        │  list[ConceptNode]
        ▼
generate_hierarchy()                      # sae_lens.synthetic.hierarchy
        │  features 0..N_hierarchical-1 → HierarchyNode trees with alpha/beta
        │  features N_hierarchical..num_features-1 → free (no HierarchyNode)
        ▼
semantic_initializer()                    # sae_lens.synthetic.feature_dictionary
        │  indices 0..N_hierarchical-1 → d_child = α·d_parent + β·d_⊥
        │  indices N_hierarchical..num_features-1 → orthogonalize_embeddings()
        ▼
FeatureDictionary                         # shape: (num_features, hidden_dim)
        │
        ▼
SyntheticModel.sample()                   # training data for SAE
```

---

## Related Pages

- [Synthetic Data](synthetic_data.md) — core synthetic data API including `FeatureDictionary`, `ActivationGenerator`, `HierarchyConfig`, and evaluation metrics
- [SynthSAEBench](synth_sae_bench.md) — standardized benchmark using the random hierarchy baseline
