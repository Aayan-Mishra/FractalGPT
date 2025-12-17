from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from fractal.models.backbones.esm2 import ESM2Backbone, ESM2BackboneConfig
from fractal.models.heads.confidence import ConfidenceHead
from fractal.models.heads.pairwise import PairwiseConstraintHead
from fractal.models.heads.torsion import TorsionAngleHead
from fractal.models.types import ConstraintPredictions


@dataclass(frozen=True)
class ConstraintPredictorConfig:
    """Configuration for the constraint prediction model.

    Notes:
    - `esm_checkpoint` is an esm.pretrained function name.
    - The ESM backbone weights are never saved inside save_pretrained().
      Only trainable heads + config are saved.
    """

    esm_checkpoint: str = "esm2_t30_150M_UR50D"
    freeze_backbone: bool = True
    unfreeze_last_n_layers: int = 0

    num_distance_bins: int = 64
    pairwise_dim: int = 128
    head_hidden_dim: int = 128
    pairwise_chunk_size: int = 64
    pairwise_checkpointing: bool = False

    predict_confidence: bool = True

    max_len: int = 1024


class ConstraintPredictor(nn.Module):
    """ESM-2 encoder + lightweight heads that predict geometric constraints.

    Critical design rule:
    - This model NEVER predicts raw XYZ coordinates.
    - It outputs structured tensors suitable for downstream deterministic folding.
    """

    def __init__(self, cfg: ConstraintPredictorConfig):
        super().__init__()
        self.cfg = cfg

        self.backbone = ESM2Backbone(
            ESM2BackboneConfig(
                checkpoint=cfg.esm_checkpoint,
                freeze=cfg.freeze_backbone,
                unfreeze_last_n_layers=cfg.unfreeze_last_n_layers,
                repr_layer=None,
            )
        )

        c = self.backbone.embedding_dim
        self.pairwise_head = PairwiseConstraintHead(
            input_dim=c,
            pairwise_dim=cfg.pairwise_dim,
            hidden_dim=cfg.head_hidden_dim,
            num_distance_bins=cfg.num_distance_bins,
            chunk_size=cfg.pairwise_chunk_size,
            checkpoint_chunks=cfg.pairwise_checkpointing,
            symmetric=True,
        )

        self.torsion_head = TorsionAngleHead(input_dim=c, hidden_dim=cfg.head_hidden_dim)

        self.confidence_head: nn.Module | None
        if cfg.predict_confidence:
            self.confidence_head = ConfidenceHead(input_dim=c, hidden_dim=cfg.head_hidden_dim)
        else:
            self.confidence_head = None

    def forward_tokens(self, tokens: torch.Tensor, residue_mask: torch.Tensor) -> dict[str, torch.Tensor | None]:
        reps = self.backbone(tokens, residue_mask=residue_mask, device=tokens.device)

        dist_logits, contact_logits = self.pairwise_head(reps, residue_mask)
        torsions = self.torsion_head(reps, residue_mask)
        confidence = self.confidence_head(reps, residue_mask) if self.confidence_head is not None else None

        return {
            "distance_logits": dist_logits,
            "contact_logits": contact_logits,
            "torsion_angles": torsions,
            "confidence": confidence,
            "residue_mask": residue_mask,
        }

    @torch.no_grad()
    def predict_from_sequence(self, sequence: str, *, device: str = "cpu") -> ConstraintPredictions:
        self.eval()
        dev = torch.device(device)

        tokens, residue_mask = self.backbone.tokenize([sequence], truncate_to=self.cfg.max_len)
        tokens = tokens.to(dev)

        out = self.forward_tokens(tokens=tokens, residue_mask=residue_mask.to(dev))

        return ConstraintPredictions(
            sequence=sequence[: self.cfg.max_len],
            distance_logits=out["distance_logits"],
            contact_logits=out["contact_logits"],
            torsion_angles=out["torsion_angles"],
            confidence=out["confidence"],
            residue_mask=out["residue_mask"],
        )

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save trainable components + config in a HF-like layout.

        Files:
          - config.json
          - pytorch_model.bin

        Backbone weights are NOT saved.
        """

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        config_path = save_dir / "config.json"
        weights_path = save_dir / "pytorch_model.bin"

        payload = asdict(self.cfg)
        payload["_class"] = self.__class__.__name__
        payload["_note"] = "Backbone weights not included; cfg.esm_checkpoint identifies backbone."

        config_path.write_text(json.dumps(payload, indent=2) + "\n")

        # Save ONLY trainable parameters (heads; optionally any explicitly-unfrozen backbone params).
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        sd = self.state_dict()
        trainable_sd = {k: v for k, v in sd.items() if k in trainable}
        torch.save({"state_dict": trainable_sd, "trainable_keys": sorted(trainable)}, weights_path)

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str | Path,
        *,
        device: str = "cpu",
        override_cfg: dict | None = None,
    ) -> "ConstraintPredictor":
        load_dir = Path(load_directory)
        cfg_path = load_dir / "config.json"
        weights_path = load_dir / "pytorch_model.bin"

        cfg_dict = json.loads(cfg_path.read_text())
        cfg_dict.pop("_class", None)
        cfg_dict.pop("_note", None)
        if override_cfg:
            cfg_dict.update(override_cfg)

        cfg = ConstraintPredictorConfig(**cfg_dict)
        model = cls(cfg)

        payload = torch.load(weights_path, map_location="cpu")
        sd = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        model.load_state_dict(sd, strict=False)
        model.to(torch.device(device))
        return model
