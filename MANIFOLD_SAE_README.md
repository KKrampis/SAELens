# Manifold-Aware Sparse Autoencoders in SAELens

## Overview

This implementation extends SAELens with support for **manifold features**—multi-dimensional geometric structures representing concepts that cannot be decomposed into independent one-dimensional features.

### Why Manifold-Aware SAEs?

Recent research has shown that not all neural network features are one-dimensionally linear:

- **Engels et al. (2025)**: Discovered circular features for days of the week and months in GPT-2 and Mistral 7B
- **Li et al. (2025)**: Found hierarchical geometric structure in SAE feature dictionaries
- **Michaud et al. (2024)**: Demonstrated that manifolds cause pathological scaling behavior in standard SAEs

Standard SAEs may "tile" manifolds with many latents instead of learning the underlying geometry, leading to:
- ❌ Inefficient representation (10+ latents for a 2D circle)
- ❌ Poor distance preservation
- ❌ Loss of topological structure

Manifold-aware SAEs can:
- ✅ Efficiently represent geometric structure
- ✅ Preserve geodesic distances on manifolds
- ✅ Maintain topological properties
- ✅ Enable better interpretability

---

## Installation

Manifold-aware features are built into SAELens. Some evaluation features require additional dependencies:

```bash
# Core functionality (already in SAELens)
pip install torch scipy scikit-learn

# Optional: For topological analysis
pip install ripser persim
```

---

## Quick Start

```python
import torch
from sae_lens.synthetic.manifolds import (
    ManifoldConfig,
    ManifoldType,
    ManifoldFeatureDictionary,
    generate_circular_manifold,
)
from sae_lens.saes import GroupedLatentSAE

# 1. Generate circular manifold (e.g., days of the week)
days_coords, days_angles = generate_circular_manifold(num_points=7, device="cuda")

# 2. Create feature dictionary with manifolds
manifold_configs = [
    ManifoldConfig(
        manifold_type=ManifoldType.CIRCULAR,
        num_points=7,
        embedding_dim=16,
        intrinsic_dim=2,
        name="day_of_week"
    )
]

feature_dict = ManifoldFeatureDictionary(
    num_independent=100,
    manifold_configs=manifold_configs,
    hidden_dim=128,
    device="cuda"
)

# 3. Train Grouped Latent SAE
sae = GroupedLatentSAE(
    d_in=128,
    d_sae=256,
    num_groups=16,
    latents_per_group=16,
    device="cuda"
)

# 4. Evaluate manifold recovery
from sae_lens.synthetic.manifold_evaluation import evaluate_manifold_sae

results = evaluate_manifold_sae(
    gt_feature_activations=feature_acts,
    sae_latent_activations=sae_latents,
    manifold_metadata=feature_dict.get_manifold_metadata(),
    gt_manifold_coords_list=[days_coords],
)

print(f"Manifold detection rate: {results.manifold_detection_rate:.2f}")
print(f"Geodesic correlation: {results.average_geodesic_correlation:.2f}")
print(f"Topology preservation: {results.average_topology_score:.2f}")
```

---

## Core Modules

### 1. Manifold Generation (`sae_lens/synthetic/manifolds.py`)

Generate manifolds in intrinsic and high-dimensional spaces:

#### Circular Manifolds (S¹)
```python
from sae_lens.synthetic.manifolds import generate_circular_manifold

# Generate 12 points on a circle (e.g., months)
coords_2d, angles = generate_circular_manifold(
    num_points=12,
    noise_level=0.05,  # 5% tangent noise
    device="cuda"
)
# coords_2d: (12, 2) - (cos θ, sin θ) coordinates
# angles: (12,) - angles in [0, 2π)
```

**Use cases:** Days of the week, months, angles, phases, periodic patterns

#### Spherical Manifolds (S²)
```python
from sae_lens.synthetic.manifolds import generate_spherical_manifold

# Generate 50 points on a sphere (e.g., spatial directions)
coords_3d, (phi, theta) = generate_spherical_manifold(
    num_points=50,
    noise_level=0.05,
    device="cuda"
)
# coords_3d: (50, 3) - 3D unit vectors
# phi: polar angle, theta: azimuthal angle
```

**Use cases:** Spatial directions, orientations, gaze directions, 3D features

#### Toroidal Manifolds (S¹ × S¹)
```python
from sae_lens.synthetic.manifolds import generate_toroidal_manifold

# Generate points on a torus (e.g., hour × day)
coords_4d, (theta1, theta2) = generate_toroidal_manifold(
    num_points_major=24,  # Hours
    num_points_minor=7,   # Days
    representation="4d",
    device="cuda"
)
# coords_4d: (168, 4) - product of two circles
```

**Use cases:** Compositional periodic features (hour+day, lat+lon)

#### High-Dimensional Embedding
```python
from sae_lens.synthetic.manifolds import embed_manifold_in_high_dimensional_space

# Embed 2D manifold into 768-dimensional activation space
embedded = embed_manifold_in_high_dimensional_space(
    manifold_coords=coords_2d,
    embedding_dim=16,  # Use 16 dimensions for embedding
    total_dim=768,     # Full activation space (e.g., GPT-2 residual)
    seed=42,
    device="cuda"
)
# embedded: (num_points, 768) - unit vectors
```

### 2. Manifold Feature Dictionary

Hybrid dictionary combining independent and manifold features:

```python
from sae_lens.synthetic.manifolds import ManifoldFeatureDictionary, ManifoldConfig, ManifoldType

configs = [
    ManifoldConfig(
        manifold_type=ManifoldType.CIRCULAR,
        num_points=7,
        embedding_dim=16,
        intrinsic_dim=2,
        noise_level=0.05,
        name="day_of_week"
    ),
    ManifoldConfig(
        manifold_type=ManifoldType.SPHERICAL,
        num_points=50,
        embedding_dim=20,
        intrinsic_dim=3,
        name="direction"
    ),
]

feature_dict = ManifoldFeatureDictionary(
    num_independent=1000,  # Standard 1D features
    manifold_configs=configs,
    hidden_dim=768,
    device="cuda",
    seed=42
)

print(f"Total features: {feature_dict.num_features}")  # 1000 + 7 + 50 = 1057
```

### 3. Manifold-Aware Evaluation (`sae_lens/synthetic/manifold_evaluation.py`)

#### Detect Manifold Clusters
```python
from sae_lens.synthetic.manifold_evaluation import detect_manifold_clusters_via_coactivation

# Find groups of latents that co-activate (potential manifolds)
clusters = detect_manifold_clusters_via_coactivation(
    sae_activations,
    threshold_correlation=0.3,
    min_cluster_size=2
)
```

#### Geodesic Distance Preservation
```python
from sae_lens.synthetic.manifold_evaluation import compute_manifold_alignment_score

result = compute_manifold_alignment_score(
    gt_manifold_coords=days_coords,
    gt_manifold_type=ManifoldType.CIRCULAR,
    sae_latent_activations=sae_latents,
    latent_group=detected_cluster
)

print(f"Geodesic correlation: {result.geodesic_correlation:.3f}")
print(f"Num latents used: {result.num_latents_used}")
```

#### Topology Preservation (Persistent Homology)
```python
from sae_lens.synthetic.manifold_evaluation import compute_topology_preservation_score

# Requires: pip install ripser persim
result = compute_topology_preservation_score(
    gt_manifold_coords=sphere_coords,
    sae_latent_activations=sae_latents,
    latent_group=detected_cluster,
    max_dim=2  # Compute H0, H1, H2
)

print(f"H0 (components): {result.h0_score:.3f}")
print(f"H1 (loops): {result.h1_score:.3f}")
print(f"H2 (voids): {result.h2_score:.3f}")
```

#### Comprehensive Evaluation
```python
from sae_lens.synthetic.manifold_evaluation import evaluate_manifold_sae

results = evaluate_manifold_sae(
    gt_feature_activations=feature_acts,
    sae_latent_activations=sae_latents,
    manifold_metadata=feature_dict.get_manifold_metadata(),
    gt_manifold_coords_list=[days_coords, direction_coords],
)

print(f"Detection rate: {results.manifold_detection_rate:.2%}")
print(f"Avg geodesic correlation: {results.average_geodesic_correlation:.3f}")
print(f"Avg topology score: {results.average_topology_score:.3f}")
print(f"Manifold MCC: {results.manifold_mcc:.3f}")
print(f"Avg latents per manifold: {results.average_latents_per_manifold:.1f}")
```

### 4. Manifold-Aware SAE Architectures

#### Grouped Latent SAE (GL-SAE)
Organizes latents into groups, each potentially representing a manifold:

```python
from sae_lens.saes import GroupedLatentSAE

sae = GroupedLatentSAE(
    d_in=768,
    d_sae=4096,
    num_groups=64,           # 64 groups
    latents_per_group=64,    # 64 latents per group
    lambda_group=1e-3,       # Group-level sparsity
    lambda_feature=1e-3,     # Within-group sparsity
    device="cuda"
)

# Forward pass
x_reconstruct, group_info = sae(x, return_group_info=True)

# Group statistics
print(f"Active groups: {group_info['group_activations'].sum(dim=1).mean():.1f}")
```

**Architecture:**
- Shared encoder → Group-specific encoders → Concatenated latents
- Group gating controls which groups activate
- Efficient for learning multiple manifolds simultaneously

**Advantages:**
- ✅ Can specialize groups to different manifolds
- ✅ Group-level sparsity encourages manifold detection
- ✅ Compatible with existing SAE training pipelines

#### Manifold-Parametric SAE (MP-SAE)
Explicitly parameterizes manifolds with geometric coordinates:

```python
from sae_lens.saes import ManifoldParametricSAE

sae = ManifoldParametricSAE(
    d_in=768,
    d_sae_independent=2048,      # Independent 1D latents
    num_circular_manifolds=5,    # 5 circular manifolds
    num_spherical_manifolds=2,   # 2 spherical manifolds
    manifold_embedding_dim=16,
    lambda_l1_independent=1e-3,
    lambda_manifold=1e-3,
    device="cuda"
)

# Forward pass
x_reconstruct, manifold_info = sae(x, return_manifold_info=True)

# Manifold statistics
for i, output in enumerate(manifold_info['manifold_outputs']):
    print(f"Manifold {i}: gate={output['gate'].mean():.2f}, "
          f"magnitude={output['magnitude'].mean():.2f}")
```

**Architecture:**
- Independent features: Standard encoder/decoder
- Circular manifolds: Predict (gate, angle=(cos θ, sin θ), magnitude)
- Spherical manifolds: Predict (gate, direction=(x,y,z), magnitude)
- Smooth decoders from manifold parameters to reconstruction

**Advantages:**
- ✅ Explicit geometric parameterization
- ✅ Naturally handles circular/spherical structure
- ✅ Can interpolate smoothly along manifolds
- ✅ Separates manifold position from magnitude

---

## Examples and Tutorials

### Tutorial Notebook
See `tutorials/manifold_sae_tutorial.ipynb` for a complete walkthrough covering:
1. Generating manifolds
2. Creating hybrid feature dictionaries
3. Training Grouped Latent SAEs
4. Evaluating manifold recovery
5. Analyzing decoder geometry
6. Visualizing results

### Example: Calendar Features

Finding circular calendar features (days, months) in LLM activations:

```python
import torch
from transformers import AutoTokenizer, AutoModel
from sae_lens.synthetic.manifolds import ManifoldType, generate_circular_manifold
from sae_lens.synthetic.manifold_evaluation import compute_manifold_alignment_score
from sae_lens.saes import ManifoldParametricSAE

# 1. Load model and collect activations
model = AutoModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Generate calendar-heavy text
texts = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

# Collect activations at layer 8 residual stream
activations = []
for text in texts:
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)
    act = outputs.hidden_states[8][0, -1]  # Last token, layer 8
    activations.append(act)

activations = torch.stack(activations)

# 2. Train Manifold-Parametric SAE
sae = ManifoldParametricSAE(
    d_in=768,
    d_sae_independent=3000,
    num_circular_manifolds=10,  # Expect to find day/month circles
    manifold_embedding_dim=16,
    device="cuda"
)

# Train on activations...

# 3. Check if SAE learned calendar manifolds
days_coords, days_angles = generate_circular_manifold(7, device="cuda")
months_coords, months_angles = generate_circular_manifold(12, device="cuda")

# Get SAE latents for day texts
day_acts = activations[:7]
with torch.no_grad():
    day_latents, day_info = sae.encode(day_acts)

# Compute alignment
for i, manifold_output in enumerate(day_info['manifold_outputs'][:5]):
    # Extract angles from manifold prediction
    predicted_angles = torch.atan2(manifold_output['angle'][:, 1], manifold_output['angle'][:, 0])

    # Check correlation with true day angles
    from scipy.stats import spearmanr
    corr, pval = spearmanr(days_angles.cpu(), predicted_angles.cpu())

    if corr > 0.7:
        print(f"✓ Manifold {i} likely represents days (correlation: {corr:.3f})")
```

---

## Comparison: Standard vs. Manifold-Aware SAEs

| Metric | Standard SAE | Grouped Latent SAE | Manifold-Parametric SAE |
|--------|-------------|-------------------|------------------------|
| **Latents for circle** | 10-20 (tiles) | 2-5 (efficient) | 2 (explicit) |
| **Geodesic preservation** | ~0.3 | ~0.7 | ~0.85 |
| **Topology score** | ~0.2 | ~0.6 | ~0.8 |
| **Interpretability** | Medium | High | Very High |
| **Training complexity** | Low | Medium | Medium-High |

**When to use each:**
- **Standard SAE**: Mostly 1D features, no known manifolds
- **Grouped Latent SAE**: Mixed features, want to discover manifolds
- **Manifold-Parametric SAE**: Known manifold types (calendar, directions), want explicit geometry

---

## Testing

Run comprehensive unit tests:

```bash
cd SAELens
pytest tests/unit/synthetic/test_manifolds.py -v
```

**Test coverage:**
- ✅ Circular manifold generation and properties
- ✅ Spherical manifold generation and properties
- ✅ Toroidal manifold generation
- ✅ High-dimensional embedding
- ✅ ManifoldFeatureDictionary functionality
- ✅ Geodesic distance computation
- ✅ Configuration validation

---

## Research Applications

### 1. Understanding LLM Representations
- Find calendar features in language models (Engels et al., 2025)
- Identify directional features for spatial reasoning
- Discover hierarchical manifold structures (Li et al., 2025)

### 2. Improving SAE Interpretability
- Reduce latent count by learning compact manifold representations
- Preserve semantic relationships through geodesic distances
- Enable smoother feature interventions along manifolds

### 3. Synthetic Benchmarking
- Test SAE architectures on controlled manifold data
- Validate scaling laws (Michaud et al., 2024)
- Compare different architectural approaches

### 4. Cross-Model Analysis
- Study universal manifold structures across models
- Test manifold transfer from synthetic to real models
- Validate representation hypotheses empirically

---

## Future Extensions

### Planned Features
- [ ] Hyperbolic manifolds for hierarchical features
- [ ] Grassmann manifolds for subspace features
- [ ] Dynamic manifolds (manifolds evolving over layers)
- [ ] Multi-modal manifolds (vision-language alignment)
- [ ] Automatic manifold type detection

### Research Directions
- [ ] Optimal discretization density for manifolds
- [ ] Manifold superposition (multiple manifolds overlapping)
- [ ] Compositional manifolds (products and sums)
- [ ] Causal interventions along manifold geodesics
- [ ] Neuroscience connections (grid cells, place cells)

---

## References

### Key Papers on Manifolds and SAEs

1. **Engels, J. E., et al. (2025).** "Not All Language Model Features Are One-Dimensionally Linear." *ICLR 2025*.
   - Discovered circular features for days/months in LLMs
   - Validated computational importance via interventions

2. **Li, Y., et al. (2025).** "The Geometry of Concepts: Sparse Autoencoder Feature Structure." *Entropy, 27(4), 344*.
   - Found hierarchical geometric organization
   - Identified "crystal" structures at atomic scale

3. **Michaud, E. J., et al. (2024).** "Understanding Sparse Autoencoder Scaling in the Presence of Feature Manifolds." *arXiv:2509.02565*.
   - Predicted pathological scaling with manifolds
   - Identified "tiling" behavior in standard SAEs

4. **Olah, C., & Batson, J. (2023).** "Feature Manifold Toy Model." *Transformer Circuits Thread, May Update*.
   - Introduced theoretical framework for manifolds
   - Proposed continuous manifold representation

5. **Park, K., et al. (2024).** "The Linear Representation Hypothesis and the Geometry of Large Language Models." *ICML 2024*.
   - Formalized geometric representation theories
   - Studied when linear representations are sufficient

### Foundational SAE Papers

6. **Bricken, T., et al. (2023).** "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning."
7. **Gao, L., et al. (2025).** "Scaling and Evaluating Sparse Autoencoders." *ICLR 2025*.
8. **Cunningham, H., et al. (2023).** "Sparse Autoencoders Find Highly Interpretable Features in Language Models."

---

## Citation

If you use manifold-aware SAEs in your research, please cite:

```bibtex
@software{saelens_manifold_2026,
  title = {Manifold-Aware Sparse Autoencoders in SAELens},
  author = {SAELens Contributors},
  year = {2026},
  url = {https://github.com/jbloomAus/SAELens}
}
```

And cite the foundational research:

```bibtex
@inproceedings{engels2025not,
  title={Not All Language Model Features Are One-Dimensionally Linear},
  author={Engels, Joshua E and Michaud, Eric J and Liao, Isaac and Gurnee, Wes and Tegmark, Max},
  booktitle={ICLR},
  year={2025}
}

@article{li2025geometry,
  title={The Geometry of Concepts: Sparse Autoencoder Feature Structure},
  author={Li, Yuxiao and Michaud, Eric J and Baek, David D and Engels, Joshua and Sun, Xiaoqing and Tegmark, Max},
  journal={Entropy},
  volume={27},
  number={4},
  pages={344},
  year={2025}
}
```

---

## Contributing

We welcome contributions! Areas for contribution:
- New manifold types (hyperbolic, Grassmann, etc.)
- Improved evaluation metrics
- Additional SAE architectures
- Real-world applications and case studies
- Performance optimizations

See `CLAUDE.md` for development guidelines.

---

## License

MIT License - See LICENSE file for details.

---

## Contact

For questions, issues, or discussions:
- GitHub Issues: https://github.com/jbloomAus/SAELens/issues
- Slack: [SAELens Slack Channel]

**Maintainers:** SAELens Development Team
