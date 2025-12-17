#!/usr/bin/env python3
"""Benchmark FRACTAL on CASP14 test set with structure quality metrics."""

import argparse
import json
import subprocess
import time
from pathlib import Path
import urllib.request
import zipfile

import torch
import numpy as np
from tqdm import tqdm

from fractal.models import ConstraintPredictor
from fractal.geometry.folding import fold_from_constraints


# CASP14 targets (88 total)
CASP14_TARGETS = [
    "T1024", "T1025", "T1026", "T1027", "T1028", "T1029", "T1030", "T1031",
    "T1032", "T1033", "T1034", "T1035", "T1036", "T1037", "T1038", "T1039",
    "T1040", "T1041", "T1042", "T1043", "T1044", "T1045", "T1046", "T1047",
    "T1048", "T1049", "T1050", "T1051", "T1052", "T1053", "T1054", "T1055",
    "T1056", "T1057", "T1058", "T1059", "T1060", "T1061", "T1062", "T1063",
    "T1064", "T1065", "T1066", "T1067", "T1068", "T1069", "T1070", "T1071",
    "T1072", "T1073", "T1074", "T1075", "T1076", "T1077", "T1078", "T1079",
    "T1080", "T1081", "T1082", "T1083", "T1084", "T1085", "T1086", "T1087",
    "T1088", "T1089", "T1090", "T1091", "T1092", "T1093", "T1094", "T1095",
    "T1096", "T1097", "T1098", "T1099", "T1100", "T1101", "T1102", "T1103",
    "T1104", "T1105", "T1106", "T1107", "T1108", "T1109", "T1110", "T1111",
]


def download_casp14_data(output_dir="data/casp14"):
    """Download CASP14 sequences and native structures."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/2] Downloading CASP14 dataset...")
    
    # Note: This is a simplified version - real CASP14 data requires manual download
    # For a working version, we'll create a download script that gets data from PDB
    sequences_file = output_path / "sequences.fasta"
    structures_dir = output_path / "structures"
    structures_dir.mkdir(exist_ok=True)
    
    print("  Creating CASP14 download script...")
    
    # Create download script
    download_script = """#!/usr/bin/env python3
import urllib.request
from pathlib import Path
import time

# CASP14 targets with their PDB codes (post-release)
CASP14_PDB_MAP = {
    "T1024": "6X2W", "T1025": "6X5X", "T1026": "6X64", "T1027": "6X7C",
    "T1028": "6X8F", "T1029": "6X8Y", "T1030": "6X91", "T1031": "6XBG",
    # Add more mappings as needed - this is a subset
}

output_dir = Path("data/casp14/structures")
output_dir.mkdir(parents=True, exist_ok=True)

sequences = {}

for target, pdb_id in CASP14_PDB_MAP.items():
    print(f"Downloading {target} ({pdb_id})...")
    
    try:
        # Download PDB file
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        pdb_path = output_dir / f"{target}.pdb"
        urllib.request.urlretrieve(pdb_url, pdb_path)
        
        # Extract sequence from PDB
        with open(pdb_path) as f:
            seq_lines = []
            for line in f:
                if line.startswith("SEQRES"):
                    seq_lines.append(line)
            
            # Parse sequence (simplified)
            aa_map = {
                'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
                'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
            }
            
            seq = ""
            for line in seq_lines:
                parts = line.split()[4:]  # Skip SEQRES, chain, etc.
                for res in parts:
                    if res in aa_map:
                        seq += aa_map[res]
            
            sequences[target] = seq
        
        time.sleep(0.5)  # Rate limit
        
    except Exception as e:
        print(f"  Failed to download {target}: {e}")

# Write sequences to FASTA
with open("data/casp14/sequences.fasta", "w") as f:
    for target, seq in sequences.items():
        f.write(f">{target}\\n{seq}\\n")

print(f"\\nDownloaded {len(sequences)} CASP14 targets")
"""
    
    download_script_path = output_path / "download_targets.py"
    download_script_path.write_text(download_script)
    
    print(f"  ✓ Download script created at {download_script_path}")
    print(f"\n  NOTE: Run 'python {download_script_path}' to download CASP14 data")
    print(f"  This will download native structures from RCSB PDB")
    
    return output_path


def calculate_tm_score(predicted_pdb, native_pdb):
    """Calculate TM-score using TMalign or TMscore."""
    try:
        # Try TMalign first (more common)
        result = subprocess.run(
            ["TMalign", predicted_pdb, native_pdb],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Parse TM-score from output
        for line in result.stdout.split('\n'):
            if 'TM-score=' in line and 'Chain_1' in line:
                tm_score = float(line.split('TM-score=')[1].split()[0])
                return tm_score
        
        return None
        
    except FileNotFoundError:
        print("  Warning: TMalign not found. Install: conda install -c bioconda tmalign")
        return None
    except Exception as e:
        print(f"  TM-score calculation failed: {e}")
        return None


def calculate_rmsd(predicted_pdb, native_pdb):
    """Calculate RMSD between predicted and native structures."""
    try:
        import biotite.structure as struc
        import biotite.structure.io.pdb as pdb
        
        # Load structures
        pred_file = pdb.PDBFile.read(predicted_pdb)
        pred_struct = pred_file.get_structure()[0]
        
        nat_file = pdb.PDBFile.read(native_pdb)
        nat_struct = nat_file.get_structure()[0]
        
        # Get CA atoms
        pred_ca = pred_struct[pred_struct.atom_name == "CA"]
        nat_ca = nat_struct[nat_struct.atom_name == "CA"]
        
        # Align and calculate RMSD
        min_len = min(len(pred_ca), len(nat_ca))
        pred_ca = pred_ca[:min_len]
        nat_ca = nat_ca[:min_len]
        
        # Superimpose
        pred_coords, transform = struc.superimpose(nat_ca, pred_ca)
        
        # Calculate RMSD
        rmsd = np.sqrt(np.mean(np.sum((pred_coords - nat_ca.coord) ** 2, axis=1)))
        
        return rmsd
        
    except Exception as e:
        print(f"  RMSD calculation failed: {e}")
        return None


def calculate_gdt_ts(predicted_pdb, native_pdb):
    """Calculate GDT-TS (Global Distance Test - Total Score)."""
    try:
        import biotite.structure as struc
        import biotite.structure.io.pdb as pdb
        
        # Load structures
        pred_file = pdb.PDBFile.read(predicted_pdb)
        pred_struct = pred_file.get_structure()[0]
        
        nat_file = pdb.PDBFile.read(native_pdb)
        nat_struct = nat_file.get_structure()[0]
        
        # Get CA atoms
        pred_ca = pred_struct[pred_struct.atom_name == "CA"]
        nat_ca = nat_struct[nat_struct.atom_name == "CA"]
        
        min_len = min(len(pred_ca), len(nat_ca))
        pred_ca = pred_ca[:min_len]
        nat_ca = nat_ca[:min_len]
        
        # Superimpose
        pred_coords, _ = struc.superimpose(nat_ca, pred_ca)
        
        # Calculate distances
        distances = np.sqrt(np.sum((pred_coords - nat_ca.coord) ** 2, axis=1))
        
        # GDT-TS: average of P1, P2, P4, P8 (percent under 1, 2, 4, 8 Å)
        thresholds = [1.0, 2.0, 4.0, 8.0]
        percentages = [(distances < t).sum() / len(distances) * 100 for t in thresholds]
        gdt_ts = np.mean(percentages)
        
        return gdt_ts
        
    except Exception as e:
        print(f"  GDT-TS calculation failed: {e}")
        return None


def benchmark_casp14(model, casp14_dir, device="cpu", output_dir="results/casp14"):
    """Run full CASP14 benchmark."""
    casp14_path = Path(casp14_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sequences_file = casp14_path / "sequences.fasta"
    structures_dir = casp14_path / "structures"
    
    if not sequences_file.exists():
        print(f"Error: {sequences_file} not found")
        print("Run the download script first!")
        return
    
    # Parse FASTA
    sequences = {}
    current_target = None
    with open(sequences_file) as f:
        for line in f:
            if line.startswith('>'):
                current_target = line[1:].strip()
                sequences[current_target] = ""
            elif current_target:
                sequences[current_target] += line.strip()
    
    print(f"\nFound {len(sequences)} CASP14 targets")
    
    results = []
    tm_scores = []
    rmsds = []
    gdt_scores = []
    inference_times = []
    folding_times = []
    
    print("\n" + "="*70)
    print("RUNNING CASP14 BENCHMARK")
    print("="*70)
    
    for target, sequence in tqdm(list(sequences.items()), desc="Processing targets"):
        target_results = {"target": target, "length": len(sequence)}
        
        # Check if native structure exists
        native_pdb = structures_dir / f"{target}.pdb"
        if not native_pdb.exists():
            print(f"  Skipping {target}: no native structure")
            continue
        
        try:
            # Predict constraints
            start = time.time()
            with torch.no_grad():
                predictions = model.predict_from_sequence(sequence, device=device)
            inference_time = time.time() - start
            inference_times.append(inference_time)
            target_results["inference_time"] = inference_time
            
            # Fold to structure
            start = time.time()
            structure = fold_from_constraints(
                predictions, 
                num_steps=1000, 
                lr=0.01, 
                device=device
            )
            folding_time = time.time() - start
            folding_times.append(folding_time)
            target_results["folding_time"] = folding_time
            
            # Save prediction
            pred_pdb = output_path / f"{target}_pred.pdb"
            structure.to_pdb(str(pred_pdb))
            
            # Calculate metrics
            tm_score = calculate_tm_score(str(pred_pdb), str(native_pdb))
            if tm_score is not None:
                tm_scores.append(tm_score)
                target_results["tm_score"] = tm_score
            
            rmsd = calculate_rmsd(str(pred_pdb), str(native_pdb))
            if rmsd is not None:
                rmsds.append(rmsd)
                target_results["rmsd"] = rmsd
            
            gdt = calculate_gdt_ts(str(pred_pdb), str(native_pdb))
            if gdt is not None:
                gdt_scores.append(gdt)
                target_results["gdt_ts"] = gdt
            
            results.append(target_results)
            
        except Exception as e:
            print(f"  Failed on {target}: {e}")
            target_results["error"] = str(e)
            results.append(target_results)
    
    # Summary statistics
    summary = {
        "total_targets": len(sequences),
        "successful_predictions": len(results),
        "avg_inference_time": np.mean(inference_times) if inference_times else 0,
        "avg_folding_time": np.mean(folding_times) if folding_times else 0,
    }
    
    if tm_scores:
        summary["tm_score"] = {
            "mean": np.mean(tm_scores),
            "median": np.median(tm_scores),
            "std": np.std(tm_scores),
            "min": np.min(tm_scores),
            "max": np.max(tm_scores),
        }
    
    if rmsds:
        summary["rmsd"] = {
            "mean": np.mean(rmsds),
            "median": np.median(rmsds),
            "std": np.std(rmsds),
            "min": np.min(rmsds),
            "max": np.max(rmsds),
        }
    
    if gdt_scores:
        summary["gdt_ts"] = {
            "mean": np.mean(gdt_scores),
            "median": np.median(gdt_scores),
            "std": np.std(gdt_scores),
            "min": np.min(gdt_scores),
            "max": np.max(gdt_scores),
        }
    
    # Save results
    output_json = output_path / "casp14_results.json"
    with open(output_json, "w") as f:
        json.dump({
            "summary": summary,
            "per_target_results": results,
        }, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("CASP14 BENCHMARK RESULTS")
    print("="*70)
    print(f"\nTargets processed: {len(results)}/{len(sequences)}")
    print(f"Avg inference time: {summary['avg_inference_time']:.2f}s")
    print(f"Avg folding time: {summary['avg_folding_time']:.2f}s")
    
    if tm_scores:
        print(f"\nTM-score: {summary['tm_score']['mean']:.4f} ± {summary['tm_score']['std']:.4f}")
        print(f"  (AlphaFold2 baseline: ~0.87)")
    
    if rmsds:
        print(f"\nRMSD: {summary['rmsd']['mean']:.2f}Å ± {summary['rmsd']['std']:.2f}Å")
        print(f"  (AlphaFold2 baseline: ~1.5Å)")
    
    if gdt_scores:
        print(f"\nGDT-TS: {summary['gdt_ts']['mean']:.2f} ± {summary['gdt_ts']['std']:.2f}")
        print(f"  (AlphaFold2 baseline: ~92.4)")
    
    print(f"\nResults saved to {output_json}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="CASP14 Benchmark for FRACTAL")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace repo or local path")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--casp14-dir", type=str, default="data/casp14")
    parser.add_argument("--output-dir", type=str, default="results/casp14")
    parser.add_argument("--download-only", action="store_true", help="Only download CASP14 data")
    args = parser.parse_args()
    
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║              FRACTAL CASP14 BENCHMARK SUITE                    ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    
    # Download CASP14 data
    casp14_path = download_casp14_data(args.casp14_dir)
    
    if args.download_only:
        print("\nDownload preparation complete!")
        print(f"Next: Run 'python {casp14_path}/download_targets.py'")
        return
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = ConstraintPredictor.from_pretrained(args.model, device=args.device)
    model.eval()
    print("✓ Model loaded")
    
    # Run benchmark
    benchmark_casp14(model, casp14_path, device=args.device, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
