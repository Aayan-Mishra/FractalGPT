---
library_name: fractal
license: apache-2.0
tags:
- protein-folding
- biology
- alphafold
- structure-prediction
- esm
pipeline_tag: feature-extraction
---

# FRACTAL 3B Protein Structure Predictor

**FRACTAL** (Framework for Representation-guided Atomic ConsTruction & ALignment) is a protein structure prediction system that combines ESM-2 language model embeddings with geometric constraint prediction and deterministic folding.

## Model Overview

This model uses the **3 billion parameter ESM-2** backbone (`esm2_t36_3B_UR50D`) with lightweight prediction heads trained to output:
- **Distance constraints** between residue pairs
- **Contact maps** for spatial proximity
- **Torsion angles** (φ, ψ, ω) for backbone geometry
- **Confidence scores** per residue

Unlike end-to-end coordinate prediction models, FRACTAL uses a two-stage pipeline:
1. **Constraint Prediction** (this model) - predicts geometric constraints from sequence
2. **Deterministic Folding** - converts constraints to 3D structure using gradient descent

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/YOUR_USERNAME/FRACTAL.git

# Or clone and install locally
git clone https://github.com/YOUR_USERNAME/FRACTAL.git
cd FRACTAL
pip install -e .
```

## Usage

### Using the Python API

```python
from fractal.models import ConstraintPredictor
from fractal.geometry.folding import fold_from_constraints

# Load the trained model
model = ConstraintPredictor.from_pretrained(
    "YOUR_USERNAME/fractal-3b",
    device="cuda"  # or "cpu"
)

# Predict constraints from sequence
sequence = "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL"
predictions = model.predict_from_sequence(sequence, device="cuda")

# Fold to 3D structure
structure = fold_from_constraints(
    predictions,
    num_steps=1000,
    lr=0.01,
    device="cuda"
)

# Save as PDB
structure.to_pdb("output.pdb")
```

### Using the CLI

```bash
# Predict and fold in one command
proteinfold fold input.fasta --checkpoint YOUR_USERNAME/fractal-3b --viz

# This generates:
# - input.pdb (3D structure)
# - input.html (interactive 3D viewer)
# - input.png (static render)
```

### Just Predict Constraints (No Folding)

```python
from fractal.models import ConstraintPredictor

model = ConstraintPredictor.from_pretrained("YOUR_USERNAME/fractal-3b")
predictions = model.predict_from_sequence("MNIFEMLR...")

# Access predictions
dist_logits = predictions.distance_logits  # [L, L, 64] distance bins
contact_logits = predictions.contact_logits  # [L, L] contact map
torsions = predictions.torsion_angles  # [L, 3] φ, ψ, ω
confidence = predictions.confidence  # [L] per-residue pLDDT
```

## Training Details

- **Backbone**: ESM-2 3B (`esm2_t36_3B_UR50D`) - frozen during training
- **Training Data**: ~1000 high-resolution PDB structures (resolution < 2.0Å)
- **Batch Size**: 1 (with 16 gradient accumulation steps)
- **Optimizer**: AdamW with learning rate 5e-5
- **Hardware**: Kaggle 2xT4 GPUs (16GB each)
- **Training Time**: ~3-4 hours

### Loss Function

Multi-task loss combining:
- Distance binned cross-entropy (64 bins, 0-20Å)
- Contact binary cross-entropy (8Å threshold)
- Torsion angle MSE with circular wrapping
- Confidence MSE (pLDDT-style)

## Model Architecture

```
Input Sequence → ESM-2 3B Encoder → Per-residue embeddings (2560-dim)
                                   ↓
                    ┌──────────────┴────────────────┐
                    ↓                               ↓
           PairwiseConstraintHead           TorsionAngleHead
           (outer product + conv)          (linear layers)
                    ↓                               ↓
          Distance + Contact Maps           φ, ψ, ω angles
```

## Limitations

- **Maximum sequence length**: 1024 residues
- **Training data**: Limited to ~1000 structures; may not generalize to all protein families
- **Folding speed**: Deterministic folding takes 30-60s for medium proteins (150-300 residues)
- **Accuracy**: Not competitive with AlphaFold2/3 on CASP benchmarks (research/educational project)

## Citation

If you use this model, please cite ESM-2:

```bibtex
@article{lin2022language,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and Verkuil, Robert and Kabeli, Ori and Shmueli, Yair and others},
  journal={Science},
  year={2022}
}
```

## License

MIT - See repository for details.

## Links

- **Repository**: https://github.com/Aayan-Mishra/FRACTAL
- **Issues**: https://github.com/Aayan-Mishra/FRACTAL/issues
