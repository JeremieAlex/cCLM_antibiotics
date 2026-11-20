"""Command-line arguments shared by the scripts."""

import argparse
import logging

FP_TYPES = ["Morgan", "Morgan6", "RDKit", "MACCS"]
DEVICES = ["auto", "cuda", "mps", "cpu"]


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--device", default="auto", choices=DEVICES)


def add_property_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("properties")
    g.add_argument("--property_columns", nargs="+", default=None,
                   help="Continuous CSV columns (scaled with a RobustScaler)")
    g.add_argument("--binary_property_columns", nargs="+", default=None, help="0/1 CSV columns")
    g.add_argument("--bin_property_columns", nargs="+", default=None,
                   help="CSV columns conditioned on as discrete bins")
    g.add_argument("--num_bins", type=int, default=10)
    g.add_argument("--bin_strategy", default="percentile", choices=["percentile", "uniform"])
    g.add_argument("--compute_properties", nargs="+", default=None,
                   help="RDKit descriptors to compute from the SMILES, e.g. LogP MW QED")
    g.add_argument("--scaler_file", default=None, help="Also save the fitted scaler here")
    g.add_argument("--premade_scaler", default=None, help="Use this fitted scaler instead of fitting one")


def add_training_args(p: argparse.ArgumentParser, num_epochs=50, lr_step=10, conditioning_dropout=0.1) -> None:
    g = p.add_argument_group("data")
    g.add_argument("--smiles_file", required=True, help="CSV with a SMILES column, or a .smi file")
    g.add_argument("--smiles_column", default="SMILES")
    g.add_argument("--output_model", required=True, help="Checkpoint path (.ckpt)")

    add_property_args(p)

    g = p.add_argument_group("scaffold")
    g.add_argument("--scaffold_column", default=None,
                   help="CSV column of scaffold SMILES, or 'auto' to extract Murcko scaffolds")
    g.add_argument("--scaffold_fp_type", default=None, choices=FP_TYPES)

    g = p.add_argument_group("conditioning dropout")
    g.add_argument("--conditioning_dropout", type=float, default=conditioning_dropout,
                   help="Chance a molecule loses all conditioning")
    g.add_argument("--property_dropout", type=float, default=None,
                   help="Chance each property column is hidden (default: --conditioning_dropout)")
    g.add_argument("--conditioning_dropout_mode", default="property", choices=["property", "example", "batch"])

    g = p.add_argument_group("augmentation")
    g.add_argument("--randomize_smiles", action="store_true", help="New random SMILES every epoch")
    g.add_argument("--augment_multiplier", type=int, default=None, help="SMILES per molecule, made once up front")
    g.add_argument("--augment_actives_multiplier", type=int, default=None,
                   help="Extra random SMILES per active (first binary column == 1)")

    g = p.add_argument_group("training")
    g.add_argument("--val_split", type=float, default=0.15)
    g.add_argument("--num_epochs", type=int, default=num_epochs)
    g.add_argument("--batch_size", type=int, default=128)
    g.add_argument("--learning_rate", type=float, default=1e-4)
    g.add_argument("--lr_step", type=int, default=lr_step, help="Multiply the LR by lr_gamma every N epochs")
    g.add_argument("--lr_gamma", type=float, default=0.5)
    g.add_argument("--clip_grad", type=float, default=1.0, help="0 disables clipping")
    g.add_argument("--patience", type=int, default=5)
    g.add_argument("--es_threshold", type=float, default=0.05)
    g.add_argument("--early_stopping_metric", default="conditional", choices=["conditional", "unconditional"])
    g.add_argument("--no_uncond_val", action="store_true", help="Skip the unconditional validation pass")
    g.add_argument("--property_ranges", nargs="+", default=None,
                   help="LOW HIGH per scalar/binary property for the per-epoch sampling check")
    g.add_argument("--sample_property_bins", nargs="+", default=None,
                   help="Bin per binned property for the per-epoch sampling check")
    g.add_argument("--sample_every", type=int, default=1)
    g.add_argument("--sample_size", type=int, default=128)
    g.add_argument("--save_every", type=int, default=5)
    g.add_argument("--num_workers", type=int, default=0)
    add_common_args(p)


def check_training_args(p: argparse.ArgumentParser, args) -> None:
    if args.no_uncond_val and args.early_stopping_metric == "unconditional":
        p.error("--no_uncond_val cannot be used with --early_stopping_metric unconditional")
    if args.augment_multiplier and args.randomize_smiles:
        logging.getLogger(__name__).warning("--augment_multiplier and --randomize_smiles are both set")
