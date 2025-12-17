from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fractal.models.constraint_predictor import ConstraintPredictor, ConstraintPredictorConfig
from fractal.training.losses import contact_bce, distance_bin_ce, torsion_sincos_mse


@dataclass(frozen=True)
class TrainConfig:
    """Enhanced training configuration with validation and checkpointing.
    
    Designed for research-grade AlphaFold-style training with proper
    model selection, learning rate scheduling, and early stopping.
    """

    model: ConstraintPredictorConfig
    lr: float = 1e-3
    weight_decay: float = 1e-2
    amp: bool = True
    grad_accum_steps: int = 1
    grad_clip: float = 0.0  # Gradient clipping max norm (0 = disabled)
    epochs: int = 1
    save_dir: str = "checkpoints"
    
    # Validation & checkpointing
    validate_every_n_epochs: int = 1
    save_every_n_epochs: int = 1
    keep_last_n_checkpoints: int = 3
    
    # Learning rate scheduling
    use_lr_scheduler: bool = True
    lr_scheduler_patience: int = 3
    lr_scheduler_factor: float = 0.5
    
    # Logging
    log_every_n_steps: int = 50
    
    # Early stopping
    early_stopping_patience: int = 10


class Trainer:
    """Enhanced trainer with validation, checkpointing, and model selection."""
    
    def __init__(self, cfg: TrainConfig, *, model: ConstraintPredictor):
        self.cfg = cfg
        self.model = model
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

        self.optim = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay),
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp) and self.device.type == "cuda")
        
        # Learning rate scheduler
        self.scheduler = None
        if cfg.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optim,
                mode="min",
                factor=float(cfg.lr_scheduler_factor),
                patience=int(cfg.lr_scheduler_patience),
            )
        
        # Training state
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.global_step = 0

    def fit(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader | None = None
    ) -> None:
        """Train the model with optional validation."""
        self.model.train()
        save_dir = Path(self.cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print(f"{'FRACTAL TRAINING':^70}")
        print("="*70)
        print(f"Model: {self.cfg.model.esm_checkpoint}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.cfg.epochs}")
        print(f"Learning Rate: {self.cfg.lr}")
        print(f"Gradient Clipping: {self.cfg.grad_clip if self.cfg.grad_clip > 0 else 'disabled'}")
        print(f"AMP: {'enabled' if self.cfg.amp else 'disabled'}")
        print("="*70 + "\n")

        for epoch in range(int(self.cfg.epochs)):
            print(f"\n┌─ Epoch {epoch+1}/{self.cfg.epochs} ──────────────────────────────────────────────────")
            
            # Training epoch
            train_loss = self._train_epoch(train_loader, epoch)
            
            # Validation
            if val_loader is not None and (epoch + 1) % int(self.cfg.validate_every_n_epochs) == 0:
                val_loss = self.validate(val_loader)
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    old_lr = self.optim.param_groups[0]['lr']
                    self.scheduler.step(val_loss)
                    new_lr = self.optim.param_groups[0]['lr']
                    if new_lr < old_lr:
                        print(f"    └─ Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
                
                # Model selection
                if val_loss < self.best_val_loss:
                    improvement = ((self.best_val_loss - val_loss) / self.best_val_loss) * 100
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    # Save best model
                    best_dir = save_dir / "best"
                    self._save_checkpoint(best_dir, epoch, is_best=True)
                    print(f"    └─ ✅ New best! Saved to {best_dir.name}/ (improved {improvement:.1f}%)")
                else:
                    self.epochs_without_improvement += 1
                    print(f"    └─ No improvement ({self.epochs_without_improvement}/{self.cfg.early_stopping_patience})")
                
                # Early stopping
                if self.epochs_without_improvement >= int(self.cfg.early_stopping_patience):
                    print(f"\n{'='*70}")
                    print(f"⚠️  Early stopping: No improvement for {self.cfg.early_stopping_patience} epochs")
                    print(f"{'='*70}\n")
                    break
            
            # Regular checkpointing
            if (epoch + 1) % int(self.cfg.save_every_n_epochs) == 0:
                checkpoint_dir = save_dir / f"epoch_{epoch:03d}"
                self._save_checkpoint(checkpoint_dir, epoch)
                self._cleanup_old_checkpoints(save_dir)
        
        print(f"\n{'='*70}")
        print(f"{'TRAINING COMPLETE':^70}")
        print(f"{'='*70}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best checkpoint: {save_dir / 'best'}")
        print(f"{'='*70}\n")

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Run one training epoch and return average loss."""
        self.model.train()
        self.optim.zero_grad(set_to_none=True)
        
        total_loss = 0.0
        total_dist = 0.0
        total_contact = 0.0
        total_torsion = 0.0
        num_batches = 0
        
        num_steps = len(train_loader)

        for batch_idx, batch in enumerate(train_loader, 1):
            tokens = batch["tokens"].to(self.device)
            residue_mask = batch["residue_mask"].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                out = self.model.forward_tokens(tokens=tokens, residue_mask=residue_mask)

                pair_mask = residue_mask[:, :, None] & residue_mask[:, None, :]

                loss_dist = distance_bin_ce(
                    logits=out["distance_logits"],
                    targets=batch["dist_targets"].to(self.device),
                    pair_mask=pair_mask,
                )
                loss_contact = contact_bce(
                    logits=out["contact_logits"],
                    targets=batch["contact_targets"].to(self.device),
                    pair_mask=pair_mask,
                )
                loss_torsion = torsion_sincos_mse(
                    pred=out["torsion_angles"],
                    target=batch["torsion_targets"].to(self.device),
                    residue_mask=residue_mask,
                )

                loss = loss_dist + loss_contact + loss_torsion
                loss = loss / float(self.cfg.grad_accum_steps)

            self.scaler.scale(loss).backward()

            if (self.global_step + 1) % int(self.cfg.grad_accum_steps) == 0:
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optim)
                
                # Gradient clipping if enabled
                if self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=float(self.cfg.grad_clip)
                    )
                
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)

            self.global_step += 1
            batch_loss = float(loss.detach().cpu()) * float(self.cfg.grad_accum_steps)
            total_loss += batch_loss
            total_dist += float(loss_dist.detach().cpu())
            total_contact += float(loss_contact.detach().cpu())
            total_torsion += float(loss_torsion.detach().cpu())
            num_batches += 1

            if self.global_step % int(self.cfg.log_every_n_steps) == 0:
                progress = batch_idx / num_steps * 100
                bar_length = 30
                filled = int(bar_length * batch_idx / num_steps)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                print(
                    f"\r[{bar}] {progress:5.1f}% │ "
                    f"Epoch {epoch+1} │ Step {self.global_step:4d} │ "
                    f"Loss: {batch_loss:.4f} ("
                    f"dist: {float(loss_dist.detach().cpu()):.3f} "
                    f"cont: {float(loss_contact.detach().cpu()):.3f} "
                    f"tors: {float(loss_torsion.detach().cpu()):.3f})",
                    end='', flush=True
                )
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_dist = total_dist / max(num_batches, 1)
        avg_contact = total_contact / max(num_batches, 1)
        avg_torsion = total_torsion / max(num_batches, 1)
        
        print()  # New line after progress bar
        print(f"    └─ Train Summary: Loss={avg_loss:.4f} (dist={avg_dist:.3f}, contact={avg_contact:.3f}, torsion={avg_torsion:.3f})")
        
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            tokens = batch["tokens"].to(self.device)
            residue_mask = batch["residue_mask"].to(self.device)

            out = self.model.forward_tokens(tokens=tokens, residue_mask=residue_mask)

            pair_mask = residue_mask[:, :, None] & residue_mask[:, None, :]

            loss_dist = distance_bin_ce(
                logits=out["distance_logits"],
                targets=batch["dist_targets"].to(self.device),
                pair_mask=pair_mask,
            )
            loss_contact = contact_bce(
                logits=out["contact_logits"],
                targets=batch["contact_targets"].to(self.device),
                pair_mask=pair_mask,
            )
            loss_torsion = torsion_sincos_mse(
                pred=out["torsion_angles"],
                target=batch["torsion_targets"].to(self.device),
                residue_mask=residue_mask,
            )

            loss = loss_dist + loss_contact + loss_torsion
            total_loss += float(loss.cpu())
            num_batches += 1

        self.model.train()
        avg_val_loss = total_loss / max(num_batches, 1)
        print(f"    └─ Val Summary:   Loss={avg_val_loss:.4f}")
        return avg_val_loss

    def _save_checkpoint(self, checkpoint_dir: Path, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint with training state."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "optimizer_state_dict": self.optim.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
        }
        
        if self.scheduler is not None:
            training_state["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(training_state, checkpoint_dir / "training_state.pt")
        
        # Save metadata
        metadata = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_val_loss": float(self.best_val_loss),
            "is_best": is_best,
        }
        (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def load_checkpoint(self, checkpoint_dir: str | Path) -> None:
        """Load checkpoint and resume training state."""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load model
        self.model = ConstraintPredictor.from_pretrained(checkpoint_dir)
        self.model.to(self.device)
        
        # Load training state
        training_state_path = checkpoint_dir / "training_state.pt"
        if training_state_path.exists():
            state = torch.load(training_state_path, map_location=self.device)
            self.optim.load_state_dict(state["optimizer_state_dict"])
            self.scaler.load_state_dict(state["scaler_state_dict"])
            self.global_step = state["global_step"]
            self.best_val_loss = state["best_val_loss"]
            self.epochs_without_improvement = state["epochs_without_improvement"]
            
            if self.scheduler is not None and "scheduler_state_dict" in state:
                self.scheduler.load_state_dict(state["scheduler_state_dict"])
            
            print(f"Resumed from epoch {state['epoch']}, step {self.global_step}")

    def _cleanup_old_checkpoints(self, save_dir: Path) -> None:
        """Keep only the last N checkpoints to save disk space."""
        epoch_dirs = sorted(
            [d for d in save_dir.glob("epoch_*") if d.is_dir()],
            key=lambda x: int(x.name.split("_")[1]),
        )
        
        # Keep last N checkpoints
        if len(epoch_dirs) > int(self.cfg.keep_last_n_checkpoints):
            for old_dir in epoch_dirs[:-int(self.cfg.keep_last_n_checkpoints)]:
                for file in old_dir.glob("*"):
                    file.unlink()
                old_dir.rmdir()
