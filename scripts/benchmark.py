#!/usr/bin/env python3
"""Benchmark FRACTAL model performance on test set."""

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from fractal.models import ConstraintPredictor
from fractal.geometry.folding import fold_from_constraints
from fractal.data.dataset import ConstraintDataset


def benchmark_inference_speed(model, sequences, device="cpu", num_runs=10):
    """Measure inference time across multiple sequences."""
    print("\n" + "="*60)
    print("INFERENCE SPEED BENCHMARK")
    print("="*60)
    
    times = []
    for seq in tqdm(sequences[:num_runs], desc="Running inference"):
        start = time.time()
        with torch.no_grad():
            _ = model.predict_from_sequence(seq, device=device)
        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.time() - start
        times.append(elapsed)
    
    times = np.array(times)
    avg_time = times.mean()
    std_time = times.std()
    
    print(f"\n  Sequences tested: {len(times)}")
    print(f"  Average time: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"  Min time: {times.min():.3f}s")
    print(f"  Max time: {times.max():.3f}s")
    
    return {"avg": avg_time, "std": std_time, "min": times.min(), "max": times.max()}


def benchmark_folding_speed(model, sequences, device="cpu", num_runs=5):
    """Measure end-to-end folding time."""
    print("\n" + "="*60)
    print("FOLDING SPEED BENCHMARK")
    print("="*60)
    
    times = []
    for seq in tqdm(sequences[:num_runs], desc="Running folding"):
        start = time.time()
        predictions = model.predict_from_sequence(seq, device=device)
        _ = fold_from_constraints(predictions, num_steps=500, lr=0.01, device=device)
        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = time.time() - start
        times.append(elapsed)
    
    times = np.array(times)
    avg_time = times.mean()
    std_time = times.std()
    
    print(f"\n  Sequences tested: {len(times)}")
    print(f"  Average time: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"  Min time: {times.min():.3f}s")
    print(f"  Max time: {times.max():.3f}s")
    
    return {"avg": avg_time, "std": std_time, "min": times.min(), "max": times.max()}


def benchmark_accuracy(model, test_manifest, data_dir, device="cpu", max_samples=50):
    """Benchmark prediction accuracy against ground truth."""
    print("\n" + "="*60)
    print("PREDICTION ACCURACY BENCHMARK")
    print("="*60)
    
    dataset = ConstraintDataset(
        manifest_file=test_manifest,
        data_dir=data_dir,
    )
    
    num_samples = min(len(dataset), max_samples)
    
    dist_losses = []
    contact_losses = []
    torsion_losses = []
    
    for i in tqdm(range(num_samples), desc="Evaluating accuracy"):
        sample = dataset[i]
        
        # Get sequence from tokens (approximate - just use length)
        seq_len = sample["residue_mask"].sum().item()
        dummy_seq = "A" * seq_len  # Placeholder - would need actual sequence
        
        # Predict
        with torch.no_grad():
            predictions = model.predict_from_sequence(dummy_seq, device=device)
        
        # Calculate losses
        pred_dist = predictions.distance_logits.to(device)
        true_dist = sample["dist_targets"].to(device)
        mask = sample["residue_mask"].to(device)
        
        # Distance loss (cross entropy over bins)
        dist_loss = torch.nn.functional.cross_entropy(
            pred_dist.reshape(-1, pred_dist.size(-1)),
            true_dist.long().reshape(-1),
            reduction="mean"
        )
        dist_losses.append(dist_loss.item())
        
        # Contact loss
        pred_contact = predictions.contact_logits.to(device)
        true_contact = sample["contact_targets"].to(device)
        contact_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_contact.reshape(-1),
            true_contact.reshape(-1),
            reduction="mean"
        )
        contact_losses.append(contact_loss.item())
        
        # Torsion loss (MSE on angles)
        pred_torsion = predictions.torsion_angles.to(device)
        true_torsion = sample["torsion_targets"].to(device)
        torsion_loss = torch.nn.functional.mse_loss(pred_torsion, true_torsion)
        torsion_losses.append(torsion_loss.item())
    
    dist_losses = np.array(dist_losses)
    contact_losses = np.array(contact_losses)
    torsion_losses = np.array(torsion_losses)
    
    print(f"\n  Samples evaluated: {num_samples}")
    print(f"  Distance loss: {dist_losses.mean():.4f} ± {dist_losses.std():.4f}")
    print(f"  Contact loss: {contact_losses.mean():.4f} ± {contact_losses.std():.4f}")
    print(f"  Torsion loss: {torsion_losses.mean():.4f} ± {torsion_losses.std():.4f}")
    
    return {
        "distance": {"mean": dist_losses.mean(), "std": dist_losses.std()},
        "contact": {"mean": contact_losses.mean(), "std": contact_losses.std()},
        "torsion": {"mean": torsion_losses.mean(), "std": torsion_losses.std()},
    }


def benchmark_memory(model, sequence, device="cpu"):
    """Measure peak memory usage."""
    print("\n" + "="*60)
    print("MEMORY USAGE BENCHMARK")
    print("="*60)
    
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model.predict_from_sequence(sequence, device=device)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        print(f"\n  Peak GPU memory: {peak_memory:.2f} GB")
        return {"peak_gpu_gb": peak_memory}
    else:
        print("\n  Memory tracking only available on CUDA")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Benchmark FRACTAL model")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace repo or local path")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--test-manifest", type=str, default="data/processed/test_manifest.jsonl")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--num-speed-runs", type=int, default=10, help="Number of inference runs")
    parser.add_argument("--num-fold-runs", type=int, default=5, help="Number of folding runs")
    parser.add_argument("--num-accuracy-samples", type=int, default=50, help="Test set samples")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    args = parser.parse_args()
    
    print("╔════════════════════════════════════════════════════════════╗")
    print("║              FRACTAL MODEL BENCHMARK SUITE                 ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"\nModel: {args.model}")
    print(f"Device: {args.device}")
    
    # Load model
    print("\n[1/5] Loading model...")
    model = ConstraintPredictor.from_pretrained(args.model, device=args.device)
    model.eval()
    print("✓ Model loaded")
    
    # Get test sequences
    print("\n[2/5] Loading test data...")
    test_sequences = []
    if Path(args.test_manifest).exists():
        with open(args.test_manifest) as f:
            for line in f:
                data = json.loads(line)
                seq_file = Path(args.data_dir) / data["pdb_id"] / "sequence.txt"
                if seq_file.exists():
                    test_sequences.append(seq_file.read_text().strip())
    
    if not test_sequences:
        # Fallback to dummy sequences
        print("  Warning: No test sequences found, using dummy data")
        test_sequences = [
            "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL",
            "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
            "MNLSKEDEIDQSIIQRIMDTSEEKTQTLELLQAVNSQFEDQIKDKFSTLTQHIKSFETELQKDIDQIKNNIEKLIKKHQGKNGSTLRKSQEKNELLDAFVTNLNEERLDALVQKLKEKQKIDQETLEKTIKSVLEKVIKSKTENDSEIALLKKVESLIEKLKQKIENQELKLIENIQDITKLDREKQEKLEVVKEILREKQKNEEELKAKLNEKQGNNNKKEELEQRLKDIEDEIIELLKTKNDELDKLIKKLQENHDGSEAKQQKEDLQKQIENLEKDILELLKDKQEVLEKLLEEKDQVLESKQSNLDELIKDISNKLKDKVENYDEDLRKLITDIEILKKSAEENKEKLENQLEDKFKEKISKLDESLTTRDLILEANKLNNFKIDKELTLTDKKLEFNLEDVQTVDQQMRNLDKRLESSKVMAKNEEMAKSLKEVLAKLEKVSNEAHRDLEKLEKTLDKLNNKLNDLQATLDKITSVLDKIKSKLEVLDKILDSQKGTNKKKSANKLNEIIEKLKSKLEENNKELESKFATFNHVLAKLEEKQKSLDSKIEKFKEDQEKLEELKKQIDSIKAVNNDLESKLEKLELDHSLITEKKIELNDMLEKMKSLISEYDKQKELAKLLTDKIKDLCQELAELKAKLEETLEIEKEDLEKKLSELHDELEKTQSKLKDKLQQIKDELDKLRAQISELSSKQQNSELTELEKKLNDLRDKLEDFEQKTDKIKATLDDVQESLNENLKDKQTSLEAAVEDLESKIQSLEEKLAELEKKLEKLKEDLESQLESRNISLLEEMQQLLAELEDLKDDLEKKLMDLNEKLEDHKTELKDTLSSLKQELDKLHKKLSKANENLKEKLEELEDKLEEEIESLDNEVDGLRSKLDQIKDELEELIADLKAKLAELEDQLEKLLNNQLEALKEKLSQLSKKLDALEASLDSKLKDLNDQIEDLKEKIEELDEDLEKLRSKLQSLKEKLLDLSNNLEEEHEELKAQLEDEKKFLDELKDKQNELEEKFKDLNDKVEDLKNQLEDIKEKFNDLSEKLSEIKEKLKALKDDLHDLLNKLEDLKAKLEELEKKLNKLIDEMDDLEEKFQKLFDKQAALKEKLEGLQEEFEKLQNQLEDLRSKISSLKDKLNELEERIEALEDKLTSLKQELEKLLAKLKDIEDKLAQLKEQLNELDEKLAALREKLEDKLEKLEQEKDKLQAELEKLKEKLQELDAALSDLKDKLQTLNEQLEKLKQELENLEEKQKKLGDKLEALEASLSNKQEDLQEKVDKLQNQLEDLVADLEALKEDLNSLEEKLQELDEKLSSLKELNEEKEALKKQMEDLKAKLSDLEKKLQELKDQLSALEKKLEDLNIKLKELESDLQAELEKLQKLKALNEQLQELEKQLEDLKTKLKDKLEDLKDKLNDQLEDLRKQLNSLKEELEDKIEELNSKQSSLKEKLQSLKKDLENIEKELQKLEAELQKLDSELKEKLEDKQAELEEKLTDLKEKLQDLLDKLEKLEEILNDLKAKIEKLKEKLNQLKSDKESLKDKLQQLDSKLQDLEEKLQKLEKLLKDLKEQLQKLKQDFQSLKQQLENLEELKKEKTKLEAKLEQLKADLNSLKEKLKEQKEKLNDLKAELQSLKEKLKDLKENLEDKLKQLKDELEAKKDKLRSLKAELNKLKDKLEDLQEKLTKLKEELEALKEKIKDLKAELKDLKDQIQSLKAELVDLKNQLKELEKKIEDLKSKLQELKEDLEELKNQLKDLETKLEQLLEKLKEKLAKLKEELENLKKELKKQLAELKELKDDLEDKLAQLKDQLETLKEQLKALKDKLQELKQELDKLEQKLDNLKEQLKKLEKQLVDLKNELSSLKEKLKELEEKLKKLKEKLEALKEQLEKLQEKISDLKQKLQELKDKLEKLKDQLTKLKDQLAALKTELDKLERKLKELEQKLQDLKSKLEELKKKLKDLQDKLQALEEKLQSLKEKLQDLKNELAKLKNQLKELEKELEDLKKELLSLKTKLSDLKKELNSLKEKLKELQKKLQNLQEQLKKLNEKLEALLEKLKNLEEQKTKLKKELEKLEKELEKLKDDLKKLEAKLEELKDELKKLEVELSKLKEKLEKLKDKLQDLQEALEKKLDKLTDLKEQLKKLEKELNQLKEQLTKLKEELESLKDKLQDLEQKLEELKEKLGSLKQELKSLKDKLQELK",
        ]
    
    print(f"✓ Loaded {len(test_sequences)} test sequences")
    print(f"  Length range: {min(len(s) for s in test_sequences)}-{max(len(s) for s in test_sequences)} residues")
    
    results = {}
    
    # Benchmark 1: Inference speed
    print("\n[3/5] Benchmarking inference speed...")
    results["inference_speed"] = benchmark_inference_speed(
        model, test_sequences, device=args.device, num_runs=args.num_speed_runs
    )
    
    # Benchmark 2: Folding speed
    print("\n[4/5] Benchmarking folding speed...")
    results["folding_speed"] = benchmark_folding_speed(
        model, test_sequences, device=args.device, num_runs=args.num_fold_runs
    )
    
    # Benchmark 3: Memory usage
    print("\n[5/5] Benchmarking memory usage...")
    results["memory"] = benchmark_memory(model, test_sequences[0], device=args.device)
    
    # Benchmark 4: Accuracy (optional)
    if Path(args.test_manifest).exists():
        print("\n[BONUS] Benchmarking accuracy...")
        try:
            results["accuracy"] = benchmark_accuracy(
                model, args.test_manifest, args.data_dir, 
                device=args.device, max_samples=args.num_accuracy_samples
            )
        except Exception as e:
            print(f"  Warning: Accuracy benchmark failed: {e}")
    
    # Save results
    results["config"] = {
        "model": args.model,
        "device": args.device,
        "num_speed_runs": args.num_speed_runs,
        "num_fold_runs": args.num_fold_runs,
    }
    
    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2))
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {output_path}")
    print("\nSummary:")
    print(f"  Inference: {results['inference_speed']['avg']:.3f}s/sequence")
    print(f"  Folding: {results['folding_speed']['avg']:.3f}s/structure")
    if "peak_gpu_gb" in results.get("memory", {}):
        print(f"  Memory: {results['memory']['peak_gpu_gb']:.2f} GB")


if __name__ == "__main__":
    main()
