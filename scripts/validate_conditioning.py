#!/usr/bin/env python
"""Check that property conditioning works.

Sweeps one property through a list of targets while the others are held fixed,
samples at each target, recomputes the properties with RDKit and reports how
close the molecules land: Pearson r between target and achieved mean, MAE per
molecule, mean bias, MAE of the held properties, validity and uniqueness. With a
scaffold it also reports how often the scaffold is kept.

Example:
    python scripts/validate_conditioning.py --model models/physchem.ckpt \\
        --property_columns LogP TPSA MW --sweep_property MW \\
        --sweep_values 300 350 400 450 500 --fixed LogP=2.5 TPSA=70 --n 200
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

from cclm.cli import FP_TYPES, add_common_args, setup_logging  # noqa: E402
from cclm.model.model import Model  # noqa: E402
from cclm.training.preprocessing import sample_property_conditions  # noqa: E402
from cclm.training.property_calculator import PROPERTIES, PropertyCalculator  # noqa: E402
from cclm.utils.chemistry import parse_generated_smiles  # noqa: E402
from cclm.utils.config import parse_with_config  # noqa: E402
from cclm.utils.device import get_device, set_seed  # noqa: E402
from cclm.utils.scaffold import (  # noqa: E402
    SCAFFOLD_LOWER_BOUNDED,
    resolve_scaffold,
    resolve_scaffold_fp,
    scaffold_fingerprint,
    scaffold_retention,
)

logger = logging.getLogger(__name__)


def parse_fixed(items):
    out = {}
    for item in items or []:
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--fixed entries must look like NAME=VALUE, got '{item}'")
        out[name.strip()] = float(value)
    return out


def check_request(model, cols, sweep, fixed):
    cc = model.conditioning_config
    if cc.num_binned_properties:
        raise ValueError("Models with binned properties are not supported by this script")
    if len(cols) != cc.num_scalar_properties + cc.num_binary_properties:
        raise ValueError(f"--property_columns must list all {cc.num_properties} conditioned properties")
    if model.property_names and model.property_names != cols:
        raise ValueError(f"--property_columns {cols} do not match the checkpoint's {model.property_names}")
    if sweep not in cols[:cc.num_scalar_properties]:
        raise ValueError(f"--sweep_property must be one of the scalar properties {cols[:cc.num_scalar_properties]}")
    if sweep not in PROPERTIES:
        raise ValueError(f"'{sweep}' cannot be recomputed with RDKit")
    if sweep in fixed:
        raise ValueError(f"'{sweep}' is swept, so it cannot also be fixed")
    unknown = sorted(set(fixed) - set(cols))
    if unknown:
        raise ValueError(f"--fixed names {unknown} are not in --property_columns")
    missing = [c for c in cols if c != sweep and c not in fixed]
    if missing:
        raise ValueError(f"No --fixed value for {missing}")


def sample_unique(model, n, batch_size, make_props, fps, temperature, max_batches=1000):
    """Up to ``n`` unique valid canonical SMILES, plus validity and uniqueness (%)."""
    found, seen = [], set()
    n_sampled = n_valid = 0
    for _ in range(max_batches):
        if len(found) >= n:
            break
        props = make_props()
        _, smiles, _ = model.sample(batch_size, properties=props, scaffold_fps=fps, temperature=temperature)
        n_sampled += len(smiles)
        for smi in smiles:
            mol = parse_generated_smiles(smi)
            if mol is None:
                continue
            n_valid += 1
            can = Chem.MolToSmiles(mol)
            if can not in seen:
                seen.add(can)
                if len(found) < n:
                    found.append(can)
    return found, 100 * n_valid / max(n_sampled, 1), 100 * len(seen) / max(n_valid, 1)


def validate(args):
    set_seed(args.seed)
    device = get_device(args.device)
    model = Model.load(args.model, device=device)
    cc = model.conditioning_config
    cols = list(args.property_columns)
    sweep = args.sweep_property
    fixed = parse_fixed(args.fixed)
    check_request(model, cols, sweep, fixed)

    measured = [c for c in cols if c in PROPERTIES]
    held = [c for c in measured if c != sweep]
    calc = PropertyCalculator(measured)

    scaffold = resolve_scaffold(args.scaffold_smiles, args.full_molecule)
    fps = None
    if scaffold:
        fp_type, fp_size = resolve_scaffold_fp(cc, args.scaffold_fp_type, args.scaffold_fp_size)
        fps = torch.tensor(scaffold_fingerprint(scaffold, fp_type, fp_size), device=device)
        fps = fps.unsqueeze(0).expand(args.batch_size, -1)
        own = dict(zip(measured, calc.compute([scaffold], progress=False)[0]))
        for name in SCAFFOLD_LOWER_BOUNDED:
            if name in own:
                targets = args.sweep_values if name == sweep else [fixed[name]]
                low = [t for t in targets if t < own[name]]
                if low:
                    logger.warning(f"{name} targets {low} are below the scaffold's own {name} ({own[name]:.1f})")

    summary, rows = [], []
    errors = {c: [] for c in measured}
    retention = []
    for target in args.sweep_values:
        targets = {c: target if c == sweep else fixed[c] for c in cols}
        specs = [(targets[c], targets[c]) for c in cols]

        def make_props():
            return sample_property_conditions(args.batch_size, specs, model.scaler, cols,
                                              cc.num_scalar_properties, device=device)[1]

        smiles, valid, unique = sample_unique(model, args.n, args.batch_size, make_props, fps, args.temperature)
        values = calc.compute(smiles, progress=False)
        achieved = values[:, measured.index(sweep)]
        stats = {"target": target, "achieved_mean": float(np.nanmean(achieved)),
                 "achieved_std": float(np.nanstd(achieved)),
                 "mae": float(np.nanmean(np.abs(achieved - target)))}
        stats["bias"] = stats["achieved_mean"] - target
        for j, c in enumerate(measured):
            err = np.abs(values[:, j] - targets[c])
            errors[c].extend(err[~np.isnan(err)].tolist())
            if c != sweep:
                stats[f"{c}_mean"] = float(np.nanmean(values[:, j]))
                stats[f"{c}_mae"] = float(np.nanmean(err))
        per_mol = {"achieved": achieved, **{f"achieved_{c}": values[:, measured.index(c)] for c in held}}
        if scaffold:
            exact, sub = scaffold_retention(smiles, scaffold)
            retention.append((exact, sub))
            stats["scaffold_exact_pct"] = 100 * float(exact.mean())
            stats["scaffold_substructure_pct"] = 100 * float(sub.mean())
            per_mol.update(scaffold_exact=exact, scaffold_substructure=sub)
        stats.update(valid_pct=valid, unique_pct=unique, n=len(smiles))
        summary.append(stats)
        rows += [{"target": target, "smiles": s, **{k: v[i] for k, v in per_mol.items()}}
                 for i, s in enumerate(smiles)]
        logger.info(f"{sweep}={target:g}: achieved {stats['achieved_mean']:.2f} ± {stats['achieved_std']:.2f}, "
                    f"MAE {stats['mae']:.2f}, valid {valid:.0f}%, unique {unique:.0f}%, n={len(smiles)}")

    sdf = pd.DataFrame(summary)
    ok = np.isfinite(sdf["achieved_mean"])
    r = (float(np.corrcoef(sdf["target"][ok], sdf["achieved_mean"][ok])[0, 1])
         if ok.sum() > 1 and np.ptp(sdf["achieved_mean"][ok]) > 0 else float("nan"))

    print(f"\nConditioning validation: {sweep}" + (f" (scaffold {scaffold})" if scaffold else ""))
    print(sdf.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    lines = [("Pearson r, target vs achieved mean", f"{r:+.3f}"),
             (f"{sweep} MAE per molecule", f"{np.mean(errors[sweep]):.2f}"),
             (f"{sweep} mean |bias|", f"{np.nanmean(np.abs(sdf['bias'])):.2f}")]
    lines += [(f"held {c}={fixed[c]:g}, MAE per molecule", f"{np.mean(errors[c]):.2f}") for c in held]
    if retention:
        lines += [("scaffold kept (Murcko match)", f"{100 * np.mean(np.concatenate([e for e, _ in retention])):.1f}%"),
                  ("scaffold kept (substructure)", f"{100 * np.mean(np.concatenate([s for _, s in retention])):.1f}%")]
    width = max(len(k) for k, _ in lines)
    for k, v in lines:
        print(f"  {k:<{width}}  {v}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    sdf.to_csv(out.with_name(f"{out.stem}_summary.csv"), index=False)
    logger.info(f"Wrote {out} and {out.stem}_summary.csv")
    return sdf, r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--property_columns", nargs="+", required=True, help="All conditioned properties, training order")
    p.add_argument("--sweep_property", required=True)
    p.add_argument("--sweep_values", type=float, nargs="+", required=True)
    p.add_argument("--fixed", nargs="+", default=None, help="NAME=VALUE for every other property")
    p.add_argument("--n", type=int, default=100, help="Unique valid molecules per target")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--scaffold_smiles", default=None)
    p.add_argument("--full_molecule", default=None)
    p.add_argument("--scaffold_fp_type", default=None, choices=FP_TYPES)
    p.add_argument("--scaffold_fp_size", type=int, default=None)
    p.add_argument("--output", default="results/conditioning_validation.csv")
    add_common_args(p)
    return validate(parse_with_config(p, argv))


if __name__ == "__main__":
    setup_logging()
    RDLogger.DisableLog("rdApp.*")
    main()
