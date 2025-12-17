from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from fractal.ssl_utils import configure_ssl_certificates


@dataclass(frozen=True)
class ESM2BackboneConfig:
    checkpoint: str
    freeze: bool = True
    unfreeze_last_n_layers: int = 0
    repr_layer: int | None = None


class ESM2Backbone(nn.Module):
    """Wrapper around facebookresearch/esm ESM-2 checkpoints.

    This module is responsible only for producing per-residue embeddings.
    It does *not* perform structure prediction.

    Loading follows the canonical ESM API:
      model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
      batch_converter = alphabet.get_batch_converter()
      results = model(tokens, repr_layers=[layer])

    The backbone is frozen by default; optional partial unfreezing is supported.
    """

    def __init__(self, cfg: ESM2BackboneConfig):
        super().__init__()
        self.cfg = cfg

        # ESM checkpoints download via urllib; ensure CA bundle is configured.
        configure_ssl_certificates()

        try:
            import esm  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "ESM dependency not installed. Install with: pip install -e '.[esm]'"
            ) from e

        if not hasattr(esm.pretrained, cfg.checkpoint):
            raise ValueError(
                f"Unknown ESM checkpoint '{cfg.checkpoint}'. "
                "Expected a function under esm.pretrained, e.g. 'esm2_t30_150M_UR50D'."
            )

        loader = getattr(esm.pretrained, cfg.checkpoint)
        model, alphabet = loader()
        model.eval()

        self.model = model
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

        self._apply_freezing()

    @property
    def embedding_dim(self) -> int:
        # ESM models typically expose embed_dim.
        return int(getattr(self.model, "embed_dim"))

    @property
    def num_layers(self) -> int:
        # fair-esm ESM2 models expose the transformer blocks as `model.layers`.
        layers = getattr(self.model, "layers", None)
        if isinstance(layers, nn.ModuleList):
            return int(len(layers))

        # Older/newer variants may store layer count on `model.args`.
        args = getattr(self.model, "args", None)
        if args is not None and hasattr(args, "layers"):
            return int(getattr(args, "layers"))

        # Last resort: try common naming.
        if hasattr(self.model, "num_layers"):
            return int(getattr(self.model, "num_layers"))

        raise AttributeError("Unable to determine ESM2 layer count")

    def _apply_freezing(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

        if not self.cfg.freeze:
            for p in self.model.parameters():
                p.requires_grad = True
            return

        n = int(self.cfg.unfreeze_last_n_layers)
        if n <= 0:
            return

        layers = getattr(self.model, "layers", None)
        if isinstance(layers, nn.ModuleList):
            for layer in list(layers)[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        # Common final layer norm used by ESM-2.
        final_ln = getattr(self.model, "emb_layer_norm_after", None)
        if isinstance(final_ln, nn.Module):
            for p in final_ln.parameters():
                p.requires_grad = True

    def tokenize(self, sequences: list[str], *, truncate_to: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize sequences into ESM tokens + residue mask.

        Returns:
            tokens: (B, T)
            residue_mask: (B, L) (True for residues, excludes BOS/EOS/pad)
        """

        data = [(f"seq{i}", s[:truncate_to] if truncate_to else s) for i, s in enumerate(sequences)]
        _, _, tokens = self.batch_converter(data)

        # For ESM: token 0 is BOS, and last non-pad token is EOS.
        # Compute lengths including special tokens.
        pad = int(self.alphabet.padding_idx)
        lens = (tokens != pad).sum(dim=1)
        max_len = int((lens - 2).max().item())

        residue_mask = torch.zeros((tokens.size(0), max_len), dtype=torch.bool)
        for i, l in enumerate(lens.tolist()):
            # residues are positions 1..l-2 in token space
            residue_mask[i, : (l - 2)] = True

        return tokens, residue_mask

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        residue_mask: torch.Tensor,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Compute per-residue embeddings.

        Args:
            tokens: (B, T) including BOS/EOS
            residue_mask: (B, L) boolean, residues only

        Returns:
            embeddings: (B, L, C)
        """

        if device is not None:
            tokens = tokens.to(device)

        # Determine which layer representation to extract.
        layer = self.cfg.repr_layer
        if layer is None:
            layer = self.num_layers

        backbone_frozen = all(not p.requires_grad for p in self.model.parameters())
        ctx = torch.no_grad() if backbone_frozen else torch.enable_grad()

        with ctx:
            results = self.model(tokens, repr_layers=[int(layer)], return_contacts=False)
            reps = results["representations"][int(layer)]  # (B, T, C)

        # Strip BOS/EOS and pad to residue_mask length.
        # Residue token span is tokens[:, 1:1+L]
        L = residue_mask.size(1)
        reps = reps[:, 1 : 1 + L, :]
        return reps
