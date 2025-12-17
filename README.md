# FRACTAL

**FRACTAL (Framework for Representation-guided Atomic ConsTruction & ALignment)** is a research-grade protein folding *system* inspired by AlphaFold-style pipelines and modern protein language models (ESM-2).

Philosophy:

- Intelligence predicts **structured geometric constraints** (distance distributions, contacts, torsions, confidence)
- Deterministic code enforces **geometry + physics-inspired priors**

This repository intentionally does **not** predict raw XYZ coordinates from the neural network.

## Recent Updates (AlphaFold-Style Enhancements)

**New capabilities:**
- ✅ **Validation loop** with automatic model selection
- ✅ **Checkpointing** with training state resumption
- ✅ **Learning rate scheduling** (ReduceLROnPlateau)
- ✅ **Early stopping** based on validation loss
- ✅ **Evaluation metrics**: RMSD, TM-score, GDT-TS, contact accuracy
- ✅ **Enhanced training** with better logging and error handling

## Repository layout

- `src/fractal/models/` : ESM-2 backbone wrappers + lightweight constraint heads
- `src/fractal/data/` : dataset formats + preprocessing utilities
- `src/fractal/geometry/` : deterministic folding/assembly engine
- `src/fractal/training/` : trainers, losses, configs (with validation & checkpointing)
- `src/fractal/inference/` : FASTA → constraints pipeline
- `src/fractal/evaluation/` : **NEW** - RMSD, TM-score, GDT-TS metrics
- `scripts/` : CLI-adjacent utilities (preprocess/train/eval/infer)
- `configs/` : YAML configs
- `tests/` : unit tests (geometry + model components)

## Quickstart

Create an environment and install:

- Minimal install: `pip install -e .`
- With ESM support: `pip install -e '.[esm]'`
- With tests: `pip install -e '.[test]'`

Run CLI help:

```bash
proteinfold --help
```

## Training with Validation

The new training pipeline supports:
- Automatic validation during training
- Best model selection based on validation loss
- Learning rate reduction on plateau
- Early stopping to prevent overfitting
- Checkpoint management (keeps last N checkpoints)

**Train from scratch:**
```bash
python scripts/train.py configs/train.yaml
```

**Resume from checkpoint:**
```bash
python scripts/train.py configs/train.yaml --resume checkpoints/best
```

**Config options** (see `configs/train.yaml`):
```yaml
trainer:
  epochs: 50
  validate_every_n_epochs: 1
  save_every_n_epochs: 5
  keep_last_n_checkpoints: 3
  use_lr_scheduler: true
  early_stopping_patience: 15
```

## Evaluation

Evaluate a trained model:
```bash
python scripts/eval.py checkpoints/best data/processed/test_manifest.jsonl -o results.json
```

Metrics computed:
- Distance prediction accuracy (bin classification)
- Contact prediction accuracy (binary classification)
- Torsion angle error (degrees)
- Structure quality (when ground truth available):
  - RMSD (Cα atoms)
  - TM-score
  - GDT-TS

## Next Steps to Reach AlphaFold Performance

1. **Scale dataset**: Download 10K+ PDB structures using `scripts/download_rcsb.py`
2. **Add MSA features**: Integrate evolutionary information via MMseqs2/HMMER
3. **Implement structure module**: Replace simple optimization with learned refinement
4. **Add template features**: Use homologous structure templates
5. **Distributed training**: Scale to multi-GPU with PyTorch DDP

> Note: the current implementation focuses on a correct modular system skeleton with proper training infrastructure. Dataset preprocessing/training loops are research-grade and ready for expansion.
