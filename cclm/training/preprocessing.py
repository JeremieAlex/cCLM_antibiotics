"""Reading and cleaning SMILES, property scaling and binning, and sampling targets."""

import csv
import gzip
import logging
import os
from typing import List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

from cclm.model.vocabulary import SMILESTokenizer, create_vocabulary
from cclm.training.property_calculator import FingerprintCalculator, PropertyCalculator
from cclm.utils.chemistry import canonicalize_and_filter, get_scaffold
from cclm.utils.scaffold import DEFAULT_SCAFFOLD_FP_TYPE

logger = logging.getLogger(__name__)

UNSPECIFIED = ("none", "*", "-", "any")


def compute_bin_edges(values: np.ndarray, num_bins: int, strategy: str = "percentile") -> List[float]:
    """``num_bins + 1`` edges: equal-count ("percentile") or equal-width ("uniform") bins."""
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("Binned property columns cannot contain NaN or inf; drop those rows first.")
    if num_bins < 2:
        raise ValueError(f"num_bins must be at least 2, got {num_bins}")
    if strategy == "percentile":
        edges = np.percentile(arr, np.linspace(0, 100, num_bins + 1))
    elif strategy == "uniform":
        edges = np.linspace(arr.min(), arr.max(), num_bins + 1)
    else:
        raise ValueError(f"bin_strategy must be 'percentile' or 'uniform', got '{strategy}'")
    interior = edges[1:-1]
    if len(np.unique(interior)) < len(interior):
        logger.warning(
            "Some bin edges are duplicated because of tied values, so the lower bins will be empty. "
            "Use fewer bins or bin_strategy='uniform'."
        )
    return [float(e) for e in edges]


def assign_bins(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Bin index in [0, num_bins - 1] for each value; out-of-range values go to the end bins."""
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("Binned property columns cannot contain NaN or inf; drop those rows first.")
    idx = np.digitize(arr, np.asarray(edges[1:-1]))
    return np.clip(idx, 0, len(edges) - 2).astype(np.float32)


def load_scaler(path: Optional[str]):
    return joblib.load(path) if path else None


def read_smiles_frame(path: str, smiles_column: str = "SMILES", delimiter: str = "\t") -> pd.DataFrame:
    """A CSV with all its columns, or the first column of a .smi/.txt file (optionally gzipped)."""
    if path.lower().endswith((".csv", ".csv.gz")):
        df = pd.read_csv(path)
        if smiles_column not in df.columns:
            raise ValueError(f"Column '{smiles_column}' not in {path} (columns: {list(df.columns)})")
        return df
    opener = gzip.open if path.endswith(".gz") else open
    smiles = []
    with opener(path, "rt") as f:
        for row in csv.reader(f, delimiter=delimiter):
            if row and row[0].strip() and not row[0].startswith("#"):
                smiles.append(row[0].strip())
    return pd.DataFrame({smiles_column: smiles})


def _scaffold_fps(df, smiles, scaffold_column, fp_type, fp_bits):
    if scaffold_column and scaffold_column in df.columns:
        scaffolds = df[scaffold_column].astype(str).tolist()
    else:
        logger.info("Extracting Murcko scaffolds")
        scaffolds = [get_scaffold(s) or s for s in smiles]
    calc = FingerprintCalculator(fp_type or DEFAULT_SCAFFOLD_FP_TYPE, fp_bits)
    return calc.compute(scaffolds)


def load_and_preprocess_csv(
    csv_path: str,
    smiles_column: str = "SMILES",
    property_columns: Optional[List[str]] = None,
    binary_property_columns: Optional[List[str]] = None,
    bin_property_columns: Optional[List[str]] = None,
    num_bins: int = 10,
    bin_strategy: str = "percentile",
    premade_bin_edges: Optional[dict] = None,
    compute_properties: Optional[List[str]] = None,
    scaffold_column: Optional[str] = None,
    scaffold_fp_type: Optional[str] = None,
    scaffold_fp_bits: Optional[int] = None,
    scaler_path: Optional[str] = None,
    scaler=None,
    max_heavy_atoms: int = 100,
    min_heavy_atoms: int = 10,
) -> dict:
    """Canonicalize and filter SMILES, build the property matrix and the vocabulary.

    Properties are ordered scalar (computed descriptors first, then
    ``property_columns``; RobustScaler-scaled), binary (unscaled) and binned
    (bin indices). Pass ``scaler`` or ``premade_bin_edges`` to reuse fitted ones.
    ``scaffold_column`` is a column of scaffold SMILES, or "auto" to extract
    Murcko scaffolds.
    """
    df = read_smiles_frame(csv_path, smiles_column)
    n_raw = len(df)
    smiles, kept = canonicalize_and_filter(df[smiles_column].astype(str).tolist(), max_heavy_atoms, min_heavy_atoms)
    df = df.iloc[kept].reset_index(drop=True)
    df[smiles_column] = smiles
    logger.info(f"{len(df):,} of {n_raw:,} molecules kept after filtering")

    scalar_cols = list(property_columns or [])
    binary_cols = list(binary_property_columns or [])
    bin_cols = list(bin_property_columns or [])

    if compute_properties:
        calc = PropertyCalculator(compute_properties)
        computed = pd.DataFrame(calc.compute(smiles), columns=calc.names)
        if computed.isna().any().any():
            logger.warning(f"{int(computed.isna().sum().sum())} computed values are NaN; filling with the median")
            computed = computed.fillna(computed.median())
        for col in calc.names:
            df[col] = computed[col].values
        scalar_cols = calc.names + [c for c in scalar_cols if c not in calc.names]

    missing = [c for c in scalar_cols + binary_cols + bin_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Property columns not found in {csv_path}: {missing}")
    overlap = set(bin_cols) & set(scalar_cols + binary_cols)
    if overlap:
        raise ValueError(f"Columns {sorted(overlap)} are listed as both binned and scalar/binary.")

    prop_cols = scalar_cols + binary_cols + bin_cols
    if prop_cols:
        df[prop_cols] = df[prop_cols].apply(pd.to_numeric, errors="coerce")
        bad = ~np.isfinite(df[prop_cols].to_numpy(dtype=np.float64)).all(axis=1)
        if bad.any():
            logger.warning(f"Dropping {int(bad.sum())} molecules with missing or non-numeric property values")
            df = df[~bad].reset_index(drop=True)
            smiles = df[smiles_column].tolist()
    if binary_cols and not df[binary_cols].isin([0, 1]).all().all():
        raise ValueError(f"Binary property columns {binary_cols} must contain only 0 and 1")

    blocks = []
    if scalar_cols:
        if scaler is None:
            scaler = RobustScaler().fit(df[scalar_cols])
            if scaler_path:
                os.makedirs(os.path.dirname(scaler_path) or ".", exist_ok=True)
                joblib.dump(scaler, scaler_path)
        blocks.append(scaler.transform(df[scalar_cols]))
    else:
        scaler = None
    if binary_cols:
        blocks.append(df[binary_cols].values)

    bin_edges, bin_occupancy = {}, {}
    if bin_cols:
        premade_bin_edges = premade_bin_edges or {}
        binned = []
        for col in bin_cols:
            edges = list(premade_bin_edges.get(col) or compute_bin_edges(df[col].values, num_bins, bin_strategy))
            idx = assign_bins(df[col].values, edges)
            counts = np.bincount(idx.astype(int), minlength=len(edges) - 1)
            logger.info(f"Bins for '{col}': occupancy {counts.tolist()}")
            if (counts == 0).any():
                logger.warning(f"Empty bins for '{col}': {np.flatnonzero(counts == 0).tolist()}")
            bin_edges[col], bin_occupancy[col] = edges, counts.tolist()
            binned.append(idx)
        if len({len(e) for e in bin_edges.values()}) > 1:
            raise ValueError("All binned properties must use the same number of bins.")
        num_bins = len(next(iter(bin_edges.values()))) - 1
        blocks.append(np.stack(binned, axis=1))

    properties = np.concatenate(blocks, axis=1).astype(np.float32) if blocks else None

    scaffold_fps = None
    if scaffold_column:
        scaffold_fps = _scaffold_fps(df, smiles, scaffold_column, scaffold_fp_type, scaffold_fp_bits)

    tokenizer = SMILESTokenizer()
    vocabulary = create_vocabulary(smiles, tokenizer)
    logger.info(f"Vocabulary: {len(vocabulary)} tokens")

    return {
        "smiles": smiles,
        "properties": properties,
        "property_names": scalar_cols + binary_cols + bin_cols,
        "num_scalar_properties": len(scalar_cols),
        "num_binary_properties": len(binary_cols),
        "num_binned_properties": len(bin_cols),
        "num_bins": num_bins if bin_cols else 0,
        "bin_edges": bin_edges,
        "bin_occupancy": bin_occupancy,
        "scaffold_fps": scaffold_fps,
        "vocabulary": vocabulary,
        "tokenizer": tokenizer,
        "scaler": scaler,
        "n_raw": n_raw,
    }


def parse_property_ranges(tokens, n_expected: int) -> list:
    """``--property_ranges`` into one (low, high) or None (unspecified) per scalar/binary property."""
    toks = [str(t) for t in (tokens or [])]
    specs, i = [], 0
    while i < len(toks):
        if toks[i].strip().lower() in UNSPECIFIED:
            specs.append(None)
            i += 1
            continue
        if i + 1 >= len(toks):
            raise ValueError(f"--property_ranges: '{toks[i]}' has no matching HIGH value")
        try:
            specs.append((float(toks[i]), float(toks[i + 1])))
        except ValueError:
            raise ValueError(f"--property_ranges: cannot read '{toks[i]} {toks[i + 1]}' as LOW HIGH") from None
        i += 2
    if len(specs) != n_expected:
        raise ValueError(f"--property_ranges describes {len(specs)} properties, expected {n_expected}")
    return specs


def parse_property_bins(tokens, num_bins: int, n_expected: int) -> list:
    """``--property_bins`` into one ("bin", lo, hi) or None per binned property.

    Each token is a bin index, an inclusive range "7-9", "top", or none/*.
    """
    specs = []
    for tok in (str(t).strip().lower() for t in (tokens or [])):
        try:
            if tok in UNSPECIFIED:
                specs.append(None)
                continue
            if tok == "top":
                lo = hi = num_bins - 1
            elif "-" in tok:
                lo, hi = (int(x) for x in tok.split("-", 1))
            else:
                lo = hi = int(float(tok))
        except ValueError:
            raise ValueError(f"--property_bins: cannot read '{tok}'") from None
        if not 0 <= lo <= hi < num_bins:
            raise ValueError(f"--property_bins: '{tok}' is outside 0..{num_bins - 1}")
        specs.append(("bin", lo, hi))
    if len(specs) != n_expected:
        raise ValueError(f"--property_bins describes {len(specs)} properties, expected {n_expected}")
    return specs


def sample_property_conditions(
    batch_size: int,
    property_specs: List[Optional[tuple]],
    scaler,
    property_names: List[str],
    num_scalar_properties: int,
    device="cpu",
) -> Tuple[np.ndarray, torch.Tensor, Optional[torch.Tensor]]:
    """Draw one property target per molecule from ``property_specs``.

    Each spec is None (unspecified), (low, high) in original units, or
    ("bin", lo, hi). Returns (raw values with NaN for unspecified, model input
    tensor, mask or None when everything is specified).
    """
    n = len(property_specs)
    raw = np.zeros((batch_size, n), dtype=np.float32)
    mask = np.ones((batch_size, n), dtype=np.float32)
    for i, spec in enumerate(property_specs):
        if spec is None:
            mask[:, i] = 0.0
        elif spec[0] == "bin":
            raw[:, i] = np.random.randint(spec[1], spec[2] + 1, size=batch_size)
        else:
            raw[:, i] = np.random.uniform(spec[0], spec[1], size=batch_size)

    values = raw.copy()
    if num_scalar_properties:
        k = num_scalar_properties
        frame = pd.DataFrame(raw[:, :k], columns=property_names[:k])
        values[:, :k] = scaler.transform(frame)

    reported = np.where(mask > 0, raw, np.nan)
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    mask_tensor = None if mask.all() else torch.tensor(mask, device=device)
    return reported, tensor, mask_tensor


def describe_specs(names: List[str], specs: List[Optional[tuple]]) -> str:
    def fmt(spec):
        if spec is None:
            return "*"
        if spec[0] == "bin":
            return f"bin {spec[1]}-{spec[2]}"
        return f"{spec[0]:g}-{spec[1]:g}"
    return ", ".join(f"{n}={fmt(s)}" for n, s in zip(names, specs))
