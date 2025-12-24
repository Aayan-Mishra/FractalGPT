# FRACTAL Architecture

This document provides detailed architecture diagrams for the FRACTAL protein structure prediction system.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FRACTAL Pipeline                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐    ┌──────────────────┐    ┌────────────────────────┐    │
│   │   Input     │    │   Constraint     │    │   Geometry Engine      │    │
│   │   FASTA     │───▶│   Predictor      │───▶│   (Deterministic)      │    │
│   │   Sequence  │    │   (Neural)       │    │                        │    │
│   └─────────────┘    └──────────────────┘    └────────────────────────┘    │
│                              │                          │                   │
│                              ▼                          ▼                   │
│                      ┌──────────────┐          ┌──────────────┐            │
│                      │  Geometric   │          │   3D         │            │
│                      │  Constraints │          │   Structure  │            │
│                      └──────────────┘          │   (PDB)      │            │
│                                                └──────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Constraint Predictor Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Constraint Predictor                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        ESM-2 Backbone                                │  │
│   │                   (Pretrained Language Model)                        │  │
│   │                                                                      │  │
│   │   Input: Amino Acid Sequence                                         │  │
│   │   Output: Per-residue Embeddings [L x D]                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      Prediction Heads                                │  │
│   ├─────────────────┬─────────────────┬─────────────────┬───────────────┤  │
│   │   Distance      │    Contact      │    Torsion      │  Confidence   │  │
│   │   Head          │    Head         │    Head         │  Head         │  │
│   │                 │                 │                 │               │  │
│   │   Pairwise      │   Binary        │   Per-residue   │  Per-residue  │  │
│   │   [L x L x B]   │   [L x L]       │   [L x 2]       │  [L]          │  │
│   │                 │                 │   (phi, psi)    │               │  │
│   └─────────────────┴─────────────────┴─────────────────┴───────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Legend:
  L = Sequence length
  D = Embedding dimension
  B = Number of distance bins
```

## Geometry Folding Engine

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Deterministic Folding Engine                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Inputs:                                                                   │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│   │  Distance   │ │   Contact   │ │   Torsion   │ │ Confidence  │          │
│   │  Logits     │ │   Logits    │ │   Angles    │ │   Scores    │          │
│   └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘          │
│          │               │               │               │                  │
│          └───────────────┴───────┬───────┴───────────────┘                  │
│                                  ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    Constraint Integration                            │  │
│   │                                                                      │  │
│   │   1. Convert distance logits to expected distances                   │  │
│   │   2. Extract contact pairs from contact probabilities               │  │
│   │   3. Weight constraints by confidence scores                        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    Iterative Optimisation                            │  │
│   │                                                                      │  │
│   │   for step in range(num_steps):                                      │
│   │       1. Compute distance violations                                 │  │
│   │       2. Compute torsion angle penalties                            │  │
│   │       3. Apply steric clash penalties                               │  │
│   │       4. Update atomic coordinates via gradient descent             │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                  │                                          │
│                                  ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                       Output Structure                               │  │
│   │                                                                      │  │
│   │   - Backbone atoms: N, CA, C, O                                      │  │
│   │   - Format: PDB file                                                 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Training Pipeline                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                      Data Preprocessing                           │     │
│   │                                                                   │     │
│   │   PDB Files ──▶ Parse Structure ──▶ Extract Labels ──▶ Manifest  │     │
│   │                                                                   │     │
│   │   Labels extracted:                                               │     │
│   │   - Pairwise Cα distances (binned)                               │     │
│   │   - Contact maps (threshold: 8Å)                                 │     │
│   │   - Backbone torsion angles (φ, ψ)                               │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                                  │                                          │
│                                  ▼                                          │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                      Training Loop                                │     │
│   │                                                                   │     │
│   │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │     │
│   │   │   Forward   │───▶│   Compute   │───▶│  Backward   │          │     │
│   │   │   Pass      │    │   Loss      │    │  Pass       │          │     │
│   │   └─────────────┘    └─────────────┘    └─────────────┘          │     │
│   │                                                                   │     │
│   │   Loss Components:                                                │     │
│   │   - Cross-entropy (distance bins)                                │     │
│   │   - Binary cross-entropy (contacts)                              │     │
│   │   - Angular loss (torsion angles)                                │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                                  │                                          │
│                                  ▼                                          │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │                      Validation & Checkpointing                   │     │
│   │                                                                   │     │
│   │   - Periodic validation on held-out set                          │     │
│   │   - Best model selection by validation loss                      │     │
│   │   - Learning rate scheduling (ReduceLROnPlateau)                 │     │
│   │   - Early stopping to prevent overfitting                        │     │
│   └──────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```
fractal/
├── models/
│   ├── esm_wrapper.py          ◄── Wraps ESM-2 backbone
│   ├── constraint_predictor.py ◄── Main prediction model
│   ├── heads.py                ◄── Distance, contact, torsion heads
│   └── types.py                ◄── Data structures
│
├── geometry/
│   ├── folding.py              ◄── Main folding algorithm
│   ├── internal_coords.py      ◄── Torsion angle handling
│   └── primitives.py           ◄── Geometric operations
│
├── training/
│   ├── trainer.py              ◄── Training loop with validation
│   ├── losses.py               ◄── Loss functions
│   └── config.py               ◄── Training configuration
│
├── inference/
│   ├── pipeline.py             ◄── FASTA → constraints
│   └── fasta.py                ◄── Sequence parsing
│
├── evaluation/
│   └── metrics.py              ◄── RMSD, TM-score, GDT-TS
│
├── data/
│   ├── dataset.py              ◄── PyTorch dataset
│   └── preprocessing.py        ◄── PDB → training labels
│
├── webui/
│   └── app.py                  ◄── FastAPI web interface
│
└── cli.py                      ◄── Command-line interface
```

## Data Flow

```
                              INFERENCE MODE
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   User Input          Model Inference           Geometry            Output  │
│   ──────────          ───────────────           ────────            ──────  │
│                                                                             │
│   FASTA file    ──▶   ESM-2 encoder    ──▶   Folding     ──▶   PDB file    │
│   (sequence)          + heads               engine            (structure)   │
│                       (constraints)         (coordinates)                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

                              TRAINING MODE
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   Ground Truth        Model Prediction          Loss              Update    │
│   ────────────        ────────────────          ────              ──────    │
│                                                                             │
│   PDB file      ──▶   Sequence input    ──▶   Compare     ──▶   Gradient   │
│   (structure)         to model              predictions        descent     │
│                       (constraints)         with labels                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
