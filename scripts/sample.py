#!/usr/bin/env python
"""Sample molecules from a trained model.

Modes: vanilla (no conditioning), property, scaffold, property_scaffold.

Property ranges are LOW HIGH per property in training order and original units;
write none or '*' instead of a pair to leave a property unspecified:
    python scripts/sample.py --model models/physchem.ckpt --mode property \\
        --property_columns LogP TPSA MW --property_ranges 2 4 none 300 450 \\
        --n 500 --output results/sample.csv

Binned properties take a bin index, a range like 7-9, top, or none:
    python scripts/sample.py --model models/binned.ckpt --mode property \\
        --bin_property_columns activity --property_bins top --n 500 --output results/top.csv

Scaffold (give the scaffold, or a molecule to take the Murcko scaffold from):
    python scripts/sample.py --model models/scaffold.ckpt --mode scaffold \\
        --full_molecule "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O" --n 500 --output results/scaf.csv
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cclm.threads import limit_threads  # noqa: E402

limit_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from tqdm import tqdm  # noqa: E402

from cclm.cli import FP_TYPES, add_common_args, setup_logging  # noqa: E402
from cclm.model.model import Model  # noqa: E402
from cclm.training.preprocessing import (  # noqa: E402
    describe_specs,
    parse_property_bins,
    parse_property_ranges,
    sample_property_conditions,
)
from cclm.utils.chemistry import parse_generated_smiles  # noqa: E402
from cclm.utils.config import parse_with_config  # noqa: E402
from cclm.utils.device import get_device, set_seed  # noqa: E402
from cclm.utils.scaffold import resolve_scaffold, resolve_scaffold_fp, scaffold_fingerprint  # noqa: E402

logger = logging.getLogger(__name__)


def check_columns(model, scalar, binary, binned):
    """The requested property columns must match the checkpoint in count, name and order."""
    cc = model.conditioning_config
    given = (len(scalar), len(binary), len(binned))
    expected = (cc.num_scalar_properties, cc.num_binary_properties, cc.num_binned_properties)
    if given != expected:
        raise ValueError(
            f"Got {given[0]} scalar / {given[1]} binary / {given[2]} binned columns, "
            f"the model expects {expected[0]} / {expected[1]} / {expected[2]}"
        )
    if model.property_names and model.property_names != scalar + binary + binned:
        raise ValueError(f"Columns {scalar + binary + binned} do not match the checkpoint's {model.property_names}")


def property_specs(args, model):
    cc = model.conditioning_config
    scalar = list(args.property_columns or [])
    binary = list(args.binary_property_columns or [])
    binned = list(args.bin_property_columns or [])
    names = scalar + binary + binned
    if not names:
        raise ValueError(f"--mode {args.mode} needs --property_columns / --binary_property_columns / "
                         "--bin_property_columns")
    check_columns(model, scalar, binary, binned)
    if cc.num_scalar_properties and model.scaler is None:
        raise ValueError("The checkpoint has no scaler")
    specs = parse_property_ranges(args.property_ranges, len(scalar) + len(binary))
    specs += parse_property_bins(args.property_bins, cc.num_bins, len(binned))
    for col, spec in zip(binned, specs[len(scalar) + len(binary):]):
        counts = model.bin_occupancy.get(col)
        if spec and counts and any(counts[b] == 0 for b in range(spec[1], spec[2] + 1)):
            logger.warning(f"Some requested bins of '{col}' had no training molecules")
    logger.info(f"Targets: {describe_specs(names, specs)}")
    return names, specs


def sample_molecules(args):
    set_seed(args.seed)
    device = get_device(args.device)
    model = Model.load(args.model, device=device)
    cc = model.conditioning_config

    use_props = args.mode in ("property", "property_scaffold")
    use_scaffold = args.mode in ("scaffold", "property_scaffold")
    if use_props and not cc.has_property_conditioning:
        raise ValueError("This model has no property conditioning")

    scaffold = resolve_scaffold(args.scaffold_smiles, args.full_molecule)
    if scaffold and not use_scaffold:
        raise ValueError(f"A scaffold was given but --mode {args.mode} does not use it")
    if use_scaffold and not scaffold:
        raise ValueError(f"--mode {args.mode} needs --scaffold_smiles or --full_molecule")

    fps = None
    if scaffold:
        fp_type, fp_size = resolve_scaffold_fp(cc, args.scaffold_fp_type, args.scaffold_fp_size)
        fp = torch.tensor(scaffold_fingerprint(scaffold, fp_type, fp_size), device=device)
        fps = fp.unsqueeze(0).expand(args.batch_size, -1)

    names, specs = property_specs(args, model) if use_props else (None, None)

    rows, seen = [], set()
    n_sampled = n_valid = 0
    bar = tqdm(total=args.n, desc="Sampling", unit="mol")
    while len(rows) < args.n:
        raw = props = mask = None
        if specs:
            raw, props, mask = sample_property_conditions(
                args.batch_size, specs, model.scaler, names, cc.num_scalar_properties, device=device
            )
        _, smiles, nlls = model.sample(args.batch_size, properties=props, scaffold_fps=fps,
                                       temperature=args.temperature, property_mask=mask)
        n_sampled += len(smiles)
        for i, (smi, nll) in enumerate(zip(smiles, nlls.tolist())):
            mol = parse_generated_smiles(smi)
            if mol is None:
                continue
            n_valid += 1
            can = Chem.MolToSmiles(mol)
            if can in seen:
                continue
            seen.add(can)
            if len(rows) >= args.n:
                continue
            row = {"smiles": can, "nll": nll}
            if raw is not None:
                row["property_seed"] = ",".join(
                    f"{n}=*" if np.isnan(v) else f"{n}={v:.4f}" for n, v in zip(names, raw[i])
                )
            if scaffold:
                row["scaffold"] = scaffold
            rows.append(row)
            bar.update(1)
        bar.set_postfix(valid=f"{100 * n_valid / n_sampled:.1f}%", unique=f"{100 * len(seen) / max(n_valid, 1):.1f}%")
    bar.close()

    df = pd.DataFrame(rows).sort_values("nll").reset_index(drop=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(df)} molecules to {args.output} "
                f"(valid {100 * n_valid / n_sampled:.1f}%, unique {100 * len(seen) / max(n_valid, 1):.1f}%)")
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--n", type=int, required=True, help="Number of unique valid molecules")
    p.add_argument("--output", required=True)
    p.add_argument("--mode", default="vanilla", choices=["vanilla", "property", "scaffold", "property_scaffold"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--property_columns", nargs="+", default=None)
    p.add_argument("--binary_property_columns", nargs="+", default=None)
    p.add_argument("--bin_property_columns", nargs="+", default=None)
    p.add_argument("--property_ranges", nargs="+", default=None,
                   help="LOW HIGH (or none) per scalar/binary property")
    p.add_argument("--property_bins", nargs="+", default=None, help="BIN, LO-HI, top or none per binned property")
    p.add_argument("--scaffold_smiles", default=None, help="Murcko scaffold to condition on")
    p.add_argument("--full_molecule", default=None, help="Molecule whose Murcko scaffold is used")
    p.add_argument("--scaffold_fp_type", default=None, choices=FP_TYPES)
    p.add_argument("--scaffold_fp_size", type=int, default=None)
    add_common_args(p)
    return sample_molecules(parse_with_config(p, argv))


if __name__ == "__main__":
    setup_logging()
    RDLogger.DisableLog("rdApp.*")
    main()
