<p align="center">
  <img src="https://img.shields.io/pypi/v/fractalml?style=flat-square" alt="PyPI Version">
  <img src="https://img.shields.io/pypi/pyversions/fractalml?style=flat-square" alt="Python Versions">
  <img src="https://img.shields.io/github/license/Aayan-Mishra/FractalGPT?style=flat-square" alt="Licence">
  <img src="https://img.shields.io/github/actions/workflow/status/Aayan-Mishra/FractalGPT/publish.yml?style=flat-square&label=build" alt="Build Status">
</p>

![Banner](https://huggingface.co/HuxleyResearch/FRACTAL-1-3B/resolve/main/banner-fractal.png)


# FRACTAL

**Framework for Representation-guided Atomic ConsTruction and ALignment**

FRACTAL is a protein structure prediction system that combines neural network-based constraint prediction with deterministic geometric folding. The architecture separates learnable components (predicting structural constraints) from rule-based assembly (enforcing physical and geometric priors).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Web Interface](#web-interface)
- [Training](#training)
- [Evaluation](#evaluation)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [Licence](#licence)

---

## Overview

FRACTAL implements a two-stage approach to protein structure prediction:

1. **Constraint Prediction**: A neural network (ESM-2 backbone with task-specific heads) predicts geometric constraints from amino acid sequences:
   - Pairwise distance distributions
   - Residue contact probabilities
   - Backbone torsion angles (phi, psi)
   - Per-residue confidence scores

2. **Deterministic Folding**: A geometry engine assembles 3D coordinates by optimising against the predicted constraints whilst enforcing physical priors (bond lengths, angles, steric clashes).

This separation ensures that neural networks focus on capturing sequence-structure relationships whilst geometric correctness is guaranteed by deterministic algorithms.

---

## Architecture

```
Input Sequence          Constraint Predictor              Geometry Engine           Output
──────────────          ────────────────────              ───────────────           ──────

                        ┌─────────────────────┐
                        │    ESM-2 Backbone   │
   FASTA         ──────▶│   (Pretrained LM)   │
   Sequence              └─────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   Prediction Heads  │
                        ├─────────┬───────────┤
                        │Distance │ Contact   │
                        │Torsion  │Confidence │
                        └────┬────┴─────┬─────┘
                             │          │
                             ▼          ▼
                        ┌─────────────────────┐          ┌─────────────┐
                        │    Constraints      │   ──────▶│   Folding   │──────▶  PDB
                        │   (L x L x B)       │          │   Engine    │       Structure
                        └─────────────────────┘          └─────────────┘
```

For detailed architecture diagrams, refer to [docs/architecture.md](docs/architecture.md).

### Constraint Predictor

The constraint predictor comprises:

| Component | Description | Output Shape |
|-----------|-------------|--------------|
| ESM-2 Backbone | Pretrained protein language model | [L, D] |
| Distance Head | Predicts binned pairwise distances | [L, L, B] |
| Contact Head | Predicts binary contact probabilities | [L, L] |
| Torsion Head | Predicts backbone dihedral angles | [L, 2] |
| Confidence Head | Estimates per-residue reliability | [L] |

Where L = sequence length, D = embedding dimension, B = number of distance bins.

### Geometry Engine

The folding engine performs iterative optimisation:

1. Initialise backbone coordinates from torsion angles
2. Compute constraint violation penalties
3. Apply steric clash corrections
4. Update coordinates via gradient descent
5. Repeat for specified iterations

---

## Installation

### From PyPI

```bash
pip install fractalml
```

### From Source

```bash
git clone https://github.com/Aayan-Mishra/FractalGPT.git
cd FractalGPT
pip install -e .
```

### Optional Dependencies

```bash
# ESM-2 language model support (required for inference)
pip install fractalml[esm]

# Web interface
pip install fractalml[webui]

# Training utilities
pip install fractalml[train]

# Data preprocessing
pip install fractalml[data]

# All dependencies
pip install fractalml[esm,webui,train,data]
```

### Requirements

- Python 3.10 or higher
- PyTorch 2.1 or higher
- CUDA-capable GPU recommended for training

---

## Quick Start

### Command Line Interface

```bash
# Display available commands
fractal --help

# Predict structure from sequence
fractal fold input.fasta --out output.pdb

# Predict with visualisation outputs
fractal fold input.fasta --out output.pdb --viz
```

### Python API

```python
from fractal.inference.pipeline import predict_constraints_from_fasta
from fractal.geometry.folding import fold_from_constraints

# Predict constraints from sequence
constraints = predict_constraints_from_fasta(
    fasta_path="protein.fasta",
    esm_checkpoint="esm2_t30_150M_UR50D",
    num_distance_bins=64,
    device="cuda",
    max_len=1024,
    checkpoint_dir="Fractal-Labs/FRACTAL-1-3B",  # HuggingFace model
)

# Fold structure from constraints
structure = fold_from_constraints(
    sequence=constraints.sequence,
    distance_logits=constraints.distance_logits,
    torsion_angles=constraints.torsion_angles,
    contact_logits=constraints.contact_logits,
    confidence=constraints.confidence,
    steps=500,
)

# Save output
structure.to_pdb("output.pdb")
```

### Using Pretrained Models

FRACTAL provides pretrained models via HuggingFace:

```bash
# Using the CLI with HuggingFace model
fractal fold input.fasta --checkpoint Fractal-Labs/FRACTAL-1-3B --out output.pdb
```

---

## Web Interface

FRACTAL includes a web-based interface for interactive structure prediction.

### Local Deployment

```bash
# Start the web server
fractal webui

# Access at http://127.0.0.1:8000
```

### Cloud Deployment (Kaggle, Colab)

```bash
# Create public URL via ngrok
fractal webui --share --ngrok-token YOUR_TOKEN

# Or set environment variable
export NGROK_AUTH_TOKEN=YOUR_TOKEN
fractal webui --share
```

Obtain a free ngrok token at: https://dashboard.ngrok.com/get-started/your-authtoken

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 8000 | Port number |
| `--reload` | True | Auto-reload on changes |
| `--share` | False | Create public URL |
| `--ngrok-token` | None | ngrok authentication token |

---

## Training

### Data Preparation

1. Download PDB structures:

```bash
python scripts/download_rcsb.py --pdb-list pdb_ids.txt --output-dir data/raw
```

2. Preprocess into training format:

```bash
python scripts/preprocess_pdb.py data/raw data/processed
```

This generates:
- `train_manifest.jsonl`: Training samples
- `val_manifest.jsonl`: Validation samples
- `test_manifest.jsonl`: Test samples

### Training Configuration

Configuration is specified via YAML files. See `configs/train.yaml`:

```yaml
model:
  esm_checkpoint: esm2_t30_150M_UR50D
  num_distance_bins: 64
  max_len: 512

trainer:
  epochs: 50
  batch_size: 4
  learning_rate: 1.0e-4
  validate_every_n_epochs: 1
  save_every_n_epochs: 5
  keep_last_n_checkpoints: 3
  use_lr_scheduler: true
  early_stopping_patience: 15

data:
  train_manifest: data/processed/train_manifest.jsonl
  val_manifest: data/processed/val_manifest.jsonl
```

### Running Training

```bash
# Train from scratch
python scripts/train.py configs/train.yaml

# Resume from checkpoint
python scripts/train.py configs/train.yaml --resume checkpoints/best
```

### Training Features

- Automatic validation with best model selection
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping based on validation loss
- Checkpoint management with configurable retention
- Comprehensive logging and error handling

---

## Evaluation

### Running Evaluation

```bash
python scripts/eval.py checkpoints/best data/processed/test_manifest.jsonl -o results.json
```

### Metrics

| Metric | Description |
|--------|-------------|
| Distance Accuracy | Bin classification accuracy for pairwise distances |
| Contact Accuracy | Binary classification accuracy for residue contacts |
| Torsion MAE | Mean absolute error for backbone angles (degrees) |
| RMSD | Root mean square deviation of Calpha atoms |
| TM-score | Template modelling score (0-1, higher is better) |
| GDT-TS | Global distance test - total score |

---

## Repository Structure

```
fractal/
├── src/fractal/
│   ├── models/              # Neural network components
│   │   ├── esm_wrapper.py   # ESM-2 backbone integration
│   │   ├── constraint_predictor.py
│   │   └── heads.py         # Prediction heads
│   │
│   ├── geometry/            # Deterministic folding
│   │   ├── folding.py       # Main folding algorithm
│   │   └── internal_coords.py
│   │
│   ├── training/            # Training infrastructure
│   │   ├── trainer.py       # Training loop
│   │   ├── losses.py        # Loss functions
│   │   └── config.py        # Configuration dataclasses
│   │
│   ├── inference/           # Inference pipeline
│   │   └── pipeline.py      # FASTA to constraints
│   │
│   ├── evaluation/          # Evaluation metrics
│   │   └── metrics.py       # RMSD, TM-score, GDT-TS
│   │
│   ├── data/                # Data handling
│   │   ├── dataset.py       # PyTorch dataset
│   │   └── preprocessing.py # PDB parsing
│   │
│   ├── webui/               # Web interface
│   │   └── app.py           # FastAPI application
│   │
│   └── cli.py               # Command-line interface
│
├── scripts/                 # Utility scripts
│   ├── train.py
│   ├── eval.py
│   ├── infer.py
│   ├── preprocess_pdb.py
│   └── download_rcsb.py
│
├── configs/                 # Configuration files
│   ├── train.yaml
│   └── infer.yaml
│
├── tests/                   # Unit tests
├── docs/                    # Documentation
└── data/                    # Data directory
```

---

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `NGROK_AUTH_TOKEN` | ngrok authentication for public URLs |
| `CUDA_VISIBLE_DEVICES` | GPU device selection |
| `HF_HOME` | HuggingFace cache directory |

### Model Checkpoints

Checkpoints are saved with the following structure:

```
checkpoints/
├── epoch_000/
│   ├── config.json          # Model configuration
│   ├── model.pt             # Model weights
│   └── optimizer.pt         # Optimiser state
├── epoch_005/
└── best/                    # Best validation checkpoint
```

---

## Roadmap

The following enhancements are planned for future releases:

1. **Dataset Scaling**: Support for larger training corpora (100K+ structures)
2. **MSA Integration**: Multiple sequence alignment features via MMseqs2/HMMER
3. **Structure Module**: Learned iterative refinement (AlphaFold-style)
4. **Template Features**: Homologous structure template integration
5. **Distributed Training**: Multi-GPU support via PyTorch DDP
6. **Side Chain Prediction**: Full atomic coordinate prediction

---

## Citation

If you use FRACTAL in your research, please cite:

```bibtex
@software{fractal2024,
  title = {FRACTAL: Framework for Representation-guided Atomic Construction and Alignment},
  author = {Mishra, Aayan},
  year = {2024},
  url = {https://github.com/Aayan-Mishra/FractalGPT}
}
```

---

## Licence

This project is licensed under the MIT Licence. See [LICENSE](LICENSE.md) for details.

---

## Acknowledgements

FRACTAL builds upon the following projects:

- [ESM-2](https://github.com/facebookresearch/esm) - Protein language models by Meta AI
- [AlphaFold](https://github.com/deepmind/alphafold) - Architectural inspiration
- [OpenFold](https://github.com/aqlaboratory/openfold) - Reference implementation

---

<p align="center">
  <strong>FRACTAL</strong> - Protein Structure Prediction
  <br>
  <a href="https://github.com/Aayan-Mishra/FractalGPT">Repository</a> |
  <a href="https://huggingface.co/Fractal-Labs/FRACTAL-1-3B">Model</a> |
  <a href="https://pypi.org/project/fractalml">PyPI</a>
</p>
