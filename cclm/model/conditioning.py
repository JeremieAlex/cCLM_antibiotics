"""Property and scaffold embedders."""

from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn

# Input value used for a hidden scalar/binary column (outside the range of real data).
NULL_SCALAR_VALUE = -10.0
NULL_BINARY_VALUE = -1.0


def _build_mlp(in_size: int, hidden_size: int, num_layers: int) -> nn.Sequential:
    layers = [nn.Linear(in_size, hidden_size), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
    layers.append(nn.Linear(hidden_size, hidden_size))
    return nn.Sequential(*layers)


@dataclass
class ConditioningConfig:
    num_scalar_properties: int = 0
    num_binary_properties: int = 0
    num_binned_properties: int = 0
    num_bins: int = 0
    scaffold_fp_size: int = 0
    scaffold_fp_type: Optional[str] = None
    condition_embedding_size: int = 256
    property_embedder_layers: int = 4
    binary_embedder_layers: int = 2
    binned_embedder_layers: int = 2
    scaffold_embedder_layers: int = 3
    conditioning_dropout: float = 0.1
    use_learned_null: bool = True

    @property
    def num_properties(self) -> int:
        return self.num_scalar_properties + self.num_binary_properties + self.num_binned_properties

    @property
    def has_property_conditioning(self) -> bool:
        return self.num_properties > 0

    @property
    def has_scaffold_conditioning(self) -> bool:
        return self.scaffold_fp_size > 0

    @property
    def total_conditioning_size(self) -> int:
        n_streams = int(self.has_property_conditioning) + int(self.has_scaffold_conditioning)
        return n_streams * self.condition_embedding_size

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ConditioningConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class BinnedPropertyHead(nn.Module):
    """Embedding table over property bins. Row ``num_bins`` of each column means "unspecified"."""

    def __init__(self, num_binned: int, num_bins: int, embedding_size: int, num_layers: int = 2):
        super().__init__()
        self.num_bins = num_bins
        stride = num_bins + 1
        self.table = nn.Embedding(num_binned * stride, embedding_size)
        self.register_buffer("_offsets", torch.arange(num_binned) * stride, persistent=False)
        self.net = _build_mlp(embedding_size, embedding_size, num_layers)

    def forward(self, bin_indices: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        idx = bin_indices.round().long().clamp(0, self.num_bins - 1)
        if mask is not None:
            idx = torch.where(mask > 0, idx, torch.full_like(idx, self.num_bins))
        emb = self.table(idx + self._offsets)
        return self.net(emb.sum(dim=1))


class MixedPropertyEmbedder(nn.Module):
    """One head per property type. Columns are ordered scalar, binary, binned.

    With more than one head the outputs are concatenated and projected back to
    ``embedding_size``.
    """

    def __init__(
        self,
        num_scalar: int,
        num_binary: int,
        embedding_size: int,
        scalar_layers: int = 4,
        binary_layers: int = 2,
        num_binned: int = 0,
        num_bins: int = 0,
        binned_layers: int = 2,
    ):
        super().__init__()
        self.num_scalar = num_scalar
        self.num_binary = num_binary
        self.num_binned = num_binned

        if num_scalar:
            self.scalar_head = _build_mlp(num_scalar, embedding_size, scalar_layers)
            self.register_buffer(
                "scalar_null_values", torch.full((num_scalar,), NULL_SCALAR_VALUE), persistent=False
            )
        if num_binary:
            self.binary_head = _build_mlp(num_binary, embedding_size, binary_layers)
            self.register_buffer(
                "binary_null_values", torch.full((num_binary,), NULL_BINARY_VALUE), persistent=False
            )
        if num_binned:
            self.binned_head = BinnedPropertyHead(num_binned, num_bins, embedding_size, binned_layers)

        n_heads = sum(n > 0 for n in (num_scalar, num_binary, num_binned))
        if n_heads > 1:
            self.fusion = nn.Linear(embedding_size * n_heads, embedding_size, bias=False)

    def forward(self, properties: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param properties: (B, num_properties)
        :param mask: (B, num_properties), 1 = visible, 0 = hidden; None means all visible
        """
        n_total = self.num_scalar + self.num_binary + self.num_binned
        if properties.dim() != 2 or properties.size(1) != n_total:
            raise ValueError(f"properties must be (batch, {n_total}), got {tuple(properties.shape)}")
        if mask is not None:
            if mask.shape != properties.shape:
                raise ValueError(f"mask shape {tuple(mask.shape)} != {tuple(properties.shape)}")
            mask = mask.to(properties.dtype)

        def cols(t, i, j):
            return None if t is None else t[:, i:j]

        def with_nulls(values, m, null):
            if m is None:
                return values
            return torch.where(m > 0, values, null.to(values.dtype).expand_as(values))

        parts = []
        i = 0
        if self.num_scalar:
            j = i + self.num_scalar
            vals = with_nulls(properties[:, i:j], cols(mask, i, j), self.scalar_null_values)
            parts.append(self.scalar_head(vals))
            i = j
        if self.num_binary:
            j = i + self.num_binary
            vals = with_nulls(properties[:, i:j], cols(mask, i, j), self.binary_null_values)
            parts.append(self.binary_head(vals))
            i = j
        if self.num_binned:
            j = i + self.num_binned
            parts.append(self.binned_head(properties[:, i:j], cols(mask, i, j)))

        if len(parts) > 1:
            return self.fusion(torch.cat(parts, dim=-1))
        return parts[0]


class ScaffoldEmbedder(nn.Module):
    def __init__(self, fp_size: int, embedding_size: int, num_layers: int = 3):
        super().__init__()
        self.net = _build_mlp(fp_size, embedding_size, num_layers)

    def forward(self, fingerprints: torch.Tensor) -> torch.Tensor:
        return self.net(fingerprints)
