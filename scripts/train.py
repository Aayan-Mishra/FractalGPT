from __future__ import annotations

from pathlib import Path

import typer
import yaml

from torch.utils.data import DataLoader

from fractal.data.dataset import PrecomputedConstraintDataset, collate_precomputed_constraints
from fractal.models.constraint_predictor import ConstraintPredictor, ConstraintPredictorConfig
from fractal.training.trainer import TrainConfig, Trainer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    config: Path = typer.Argument(..., help="Path to YAML training config."),
    resume_from: Path | None = typer.Option(None, "--resume", help="Resume from checkpoint directory."),
):
    """Train FRACTAL constraint predictor with validation and checkpointing."""
    cfg_dict = yaml.safe_load(config.read_text())

    # Model configuration
    model_cfg = ConstraintPredictorConfig(**cfg_dict["model"])
    
    # Training configuration with all new features
    trainer_cfg = cfg_dict.get("trainer", {})
    optim_cfg = cfg_dict.get("optim", {})
    
    train_cfg = TrainConfig(
        model=model_cfg,
        lr=float(optim_cfg.get("lr", 1e-3)),
        weight_decay=float(optim_cfg.get("weight_decay", 1e-2)),
        grad_accum_steps=int(optim_cfg.get("grad_accum_steps", 1)),
        grad_clip=float(optim_cfg.get("grad_clip", 0.0)),
        amp=bool(optim_cfg.get("amp", True)),
        epochs=int(trainer_cfg.get("epochs", 10)),
        save_dir=str(trainer_cfg.get("save_dir", "checkpoints")),
        # New features
        validate_every_n_epochs=int(trainer_cfg.get("validate_every_n_epochs", 1)),
        save_every_n_epochs=int(trainer_cfg.get("save_every_n_epochs", 1)),
        keep_last_n_checkpoints=int(trainer_cfg.get("keep_last_n_checkpoints", 3)),
        use_lr_scheduler=bool(trainer_cfg.get("use_lr_scheduler", True)),
        lr_scheduler_patience=int(trainer_cfg.get("lr_scheduler_patience", 3)),
        lr_scheduler_factor=float(trainer_cfg.get("lr_scheduler_factor", 0.5)),
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 50)),
        early_stopping_patience=int(trainer_cfg.get("early_stopping_patience", 10)),
    )

    # Data loaders
    data_cfg = cfg_dict.get("data", {})
    batch_size = int(data_cfg.get("batch_size", 1))
    num_workers = int(data_cfg.get("num_workers", 0))
    
    train_ds = PrecomputedConstraintDataset(data_cfg["train_manifest"])
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_precomputed_constraints,
    )
    
    # Optional validation loader
    val_loader = None
    if "val_manifest" in data_cfg:
        val_ds = PrecomputedConstraintDataset(data_cfg["val_manifest"])
        val_loader = DataLoader(
            val_ds, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_precomputed_constraints,
        )

    # Initialize model and trainer
    model = ConstraintPredictor(train_cfg.model)
    trainer = Trainer(train_cfg, model=model)
    
    # Resume from checkpoint if specified
    if resume_from is not None:
        trainer.load_checkpoint(resume_from)
    
    # Train
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    app()
