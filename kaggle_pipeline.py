#!/usr/bin/env python3
"""
FRACTAL End-to-End Pipeline for Kaggle (2xT4 GPUs)
===================================================
Downloads → Preprocesses → Trains → Inferences → Pushes to HuggingFace

Usage:
    python kaggle_pipeline.py --n-samples 1000 --hf-repo your-username/fractal-3b
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: str, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    print(f"$ {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        sys.exit(1)
    print(f"\n✅ DONE: {description}")


def main():
    parser = argparse.ArgumentParser(description="FRACTAL Kaggle Pipeline")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of PDB structures to download")
    parser.add_argument("--hf-repo", type=str, help="HuggingFace repo (e.g., username/fractal-3b)")
    parser.add_argument("--skip-download", action="store_true", help="Skip download if data exists")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing if data exists")
    parser.add_argument("--skip-train", action="store_true", help="Skip training (use existing checkpoint)")
    
    args = parser.parse_args()
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                 FRACTAL KAGGLE PIPELINE                        ║
    ║              AlphaFold-Style Protein Folding                   ║
    ║                   ESM2-3B on 2xT4 GPUs                         ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Check GPU availability
    run_cmd(
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
        "Check GPU availability"
    )
    
    # Step 1: Fetch PDB IDs
    if not args.skip_download:
        run_cmd(
            f"python scripts/fetch_pdb_ids.py --out pdb_ids_{args.n_samples}.txt --limit {args.n_samples}",
            f"Fetch {args.n_samples} PDB IDs from RCSB"
        )
        
        # Step 2: Download structures
        run_cmd(
            f"python scripts/download_rcsb.py pdb_ids_{args.n_samples}.txt --out-dir data/raw/rcsb --workers 8",
            "Download PDB structures (parallel)"
        )
    
    # Step 3: Preprocess
    if not args.skip_preprocess:
        run_cmd(
            f"python scripts/preprocess_pdb.py data/raw/rcsb --out-dir data/processed_{args.n_samples} "
            f"--esm-checkpoint esm2_t6_8M_UR50D",  # Use small tokenizer for preprocessing
            "Preprocess structures → tensors"
        )
        
        # Update config to point to new manifests
        import yaml
        config_path = Path("configs/train.yaml")
        config = yaml.safe_load(config_path.read_text())
        config["data"]["train_manifest"] = f"data/processed_{args.n_samples}/train_manifest.jsonl"
        config["data"]["val_manifest"] = f"data/processed_{args.n_samples}/val_manifest.jsonl"
        config_path.write_text(yaml.dump(config))
        print(f"✓ Updated config to use processed_{args.n_samples} data")
    
    # Step 4: Train
    if not args.skip_train:
        run_cmd(
            "python scripts/train.py configs/train.yaml",
            "Train FRACTAL model (ESM2-3B + constraint heads)"
        )
    
    # Step 5: Inference examples
    print("\n" + "="*70)
    print("STEP: Generate example predictions")
    print("="*70)
    
    examples = [
        (">insulin|Short peptide hormone", "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT"),
        (">myoglobin|Oxygen-binding protein", "GLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDKFK"),
        (">ubiquitin|Protein degradation", "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"),
    ]
    
    for i, (header, seq) in enumerate(examples, 1):
        fasta_path = Path(f"example_{i}.fasta")
        fasta_path.write_text(f"{header}\n{seq}\n")
        
        run_cmd(
            f"fractal fold {fasta_path} --checkpoint models/trained/best --device cuda",
            f"Inference example {i}: {header.split('|')[0]}"
        )
    
    print("\n✓ Generated 3 example structures (PDB + HTML + PNG)")
    
    # Step 6: Push to HuggingFace
    if args.hf_repo:
        print("\n" + "="*70)
        print(f"STEP: Push model to HuggingFace: {args.hf_repo}")
        print("="*70)
        
        # Create README
        readme_content = f"""---
license: mit
tags:
- protein-folding
- biology
- esm2
- alphafold
datasets:
- rcsb-pdb
---

# FRACTAL Protein Folding Model (ESM2-3B)

AlphaFold-style protein structure prediction using ESM-2 (3B parameters) backbone with constraint prediction heads.

## Model Description

- **Backbone**: ESM2-3B (3 billion parameters, frozen)
- **Task**: Predict distance, contact, and torsion constraints
- **Training**: {args.n_samples} PDB structures from RCSB
- **Hardware**: 2xT4 GPUs on Kaggle

## Usage

```python
from fractal.inference.pipeline import predict_constraints_from_fasta
from fractal.geometry.folding import fold_from_constraints

# Predict constraints
constraints = predict_constraints_from_fasta(
    fasta_path="protein.fasta",
    checkpoint_dir="models/trained/best",
    device="cuda"
)

# Fold to 3D structure
structure = fold_from_constraints(
    sequence=constraints.sequence,
    distance_logits=constraints.distance_logits,
    torsion_angles=constraints.torsion_angles,
    contact_logits=constraints.contact_logits,
    steps=500
)

structure.to_pdb("output.pdb")
```

## Training Details

See config at `configs/train.yaml` for hyperparameters.

## Examples

See example predictions in the repository.
"""
        
        Path("README_HF.md").write_text(readme_content)
        
        # Copy model card
        import shutil
        if Path("hf_model_card.md").exists():
            shutil.copy("hf_model_card.md", "MODEL_CARD.md")
        
        # Install huggingface_hub if needed
        run_cmd("pip install -q huggingface_hub", "Install HuggingFace Hub")
        
        # Push to hub
        push_script = f"""
from huggingface_hub import HfApi, create_repo
import os
import shutil

api = HfApi()

# Create repo (will skip if exists)
try:
    create_repo("{args.hf_repo}", repo_type="model", exist_ok=True)
except Exception as e:
    print(f"Repo creation: {{e}}")

# Prepare upload directory
upload_dir = "hf_upload"
os.makedirs(upload_dir, exist_ok=True)

# Copy checkpoint files
for f in ["config.json", "pytorch_model.bin"]:
    src = f"models/trained/best/{{f}}"
    if os.path.exists(src):
        shutil.copy(src, upload_dir)

# Copy source code for custom loading
src_fractal = "src/fractal"
dest_fractal = f"{{upload_dir}}/fractal"
if os.path.exists(dest_fractal):
    shutil.rmtree(dest_fractal)
shutil.copytree(src_fractal, dest_fractal)

# Create __init__.py for auto-loading
with open(f"{{upload_dir}}/__init__.py", "w") as f:
    f.write(\"\"\"from fractal.models import ConstraintPredictor
__all__ = ["ConstraintPredictor"]
\"\"\")

# Create requirements.txt
with open(f"{{upload_dir}}/requirements.txt", "w") as f:
    f.write(\"\"\"torch>=2.0.0
fair-esm>=2.0.0
biotite>=0.38.0
\"\"\")

# Upload entire folder
api.upload_folder(
    folder_path=upload_dir,
    repo_id="{args.hf_repo}",
    commit_message="Upload FRACTAL model with source code",
)

# Upload examples
for i in range(1, 4):
    for ext in ['fasta', 'pdb', 'html', 'png']:
        try:
            api.upload_file(
                path_or_fileobj=f"example_{{i}}.{{ext}}",
                path_in_repo=f"examples/example_{{i}}.{{ext}}",
                repo_id="{args.hf_repo}",
            )
        except Exception as e:
            print(f"Upload example_{{i}}.{{ext}}: {{e}}")

# Upload model card as README
if os.path.exists("MODEL_CARD.md"):
    api.upload_file(
        path_or_fileobj="MODEL_CARD.md",
        path_in_repo="README.md",
        repo_id="{args.hf_repo}",
    )
else:
    # Fallback to generated README
    api.upload_file(
        path_or_fileobj="README_HF.md",
        path_in_repo="README.md",
        repo_id="{args.hf_repo}",
    )

print("✓ Model pushed to HuggingFace: https://huggingface.co/{args.hf_repo}")
"""
        
        Path("push_to_hf.py").write_text(push_script)
        run_cmd("python push_to_hf.py", "Push to HuggingFace Hub")
    
    print(f"""
    
    ╔════════════════════════════════════════════════════════════════╗
    ║                   PIPELINE COMPLETE! 🎉                        ║
    ╚════════════════════════════════════════════════════════════════╝
    
    Summary:
    --------
    ✓ Downloaded {args.n_samples} PDB structures
    ✓ Preprocessed with ESM tokenization
    ✓ Trained ESM2-3B constraint predictor
    ✓ Generated 3 example predictions
    {"✓ Pushed to HuggingFace: " + args.hf_repo if args.hf_repo else ""}
    
    Next steps:
    -----------
    - Check training metrics in models/trained/best/metadata.json
    - View examples in browser (*.html files)
    - Run inference: fractal fold your_protein.fasta
    """)


if __name__ == "__main__":
    main()
