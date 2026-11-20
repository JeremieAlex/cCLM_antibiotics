"""Conditioning dropout masks.

``conditioning_dropout`` is the chance a molecule loses all of its conditioning,
which trains the null embedding used for unconditional sampling.
``property_dropout`` is the chance a single property column (or the scaffold)
is hidden, so the model accepts partially specified targets.
"""

from typing import Optional

import torch

MODES = ("property", "example", "batch")


def build_conditioning_masks(
    batch_size: int,
    num_properties: int,
    conditioning_dropout: float,
    property_dropout: Optional[float] = None,
    has_scaffold: bool = False,
    mode: str = "property",
    device=None,
):
    """Returns (property_mask (B, P) or None, scaffold_mask (B, 1) or None); 1 = visible.

    ``mode``: "property" hides columns independently, "example" drops the whole
    vector per molecule, "batch" makes a single draw for the whole batch.
    """
    if mode not in MODES:
        raise ValueError(f"conditioning_dropout_mode must be one of {MODES}, got '{mode}'")
    if property_dropout is None:
        property_dropout = conditioning_dropout

    if mode == "batch":
        drop_all = torch.full((batch_size, 1), bool(torch.rand(()) < conditioning_dropout), device=device)
    else:
        drop_all = torch.rand(batch_size, 1, device=device) < conditioning_dropout

    def mask(width):
        if mode == "property" and property_dropout > 0:
            visible = torch.rand(batch_size, width, device=device) >= property_dropout
        else:
            visible = torch.ones(batch_size, width, dtype=torch.bool, device=device)
        return (visible & ~drop_all).float()

    prop_mask = mask(num_properties) if num_properties > 0 else None
    scaf_mask = mask(1) if has_scaffold else None
    return prop_mask, scaf_mask
