#!/usr/bin/env python
"""Finetune a pretrained conditional model on a (smaller) labeled dataset.

The architecture, vocabulary, scaler and bin edges come from the checkpoint, so
use the same property columns, in the same order, as the pretraining run.

Example:
    python scripts/finetune_conditional.py --pretrained_model models/prior.ckpt \\
        --smiles_file data/labeled.csv --output_model models/finetuned.ckpt \\
        --property_columns activity --freeze_backbone --unfreeze_epoch 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cclm.threads import limit_threads  # noqa: E402

limit_threads()

from cclm.cli import add_training_args, check_training_args, setup_logging  # noqa: E402
from cclm.training.finetune import finetune_conditional  # noqa: E402
from cclm.utils.config import parse_with_config  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pretrained_model", required=True)
    add_training_args(p, num_epochs=20, lr_step=5, conditioning_dropout=0.05)
    p.add_argument("--scaffold_fp_size", type=int, default=None, help="Must match the checkpoint if given")

    g = p.add_argument_group("freezing")
    g.add_argument("--freeze_backbone", action="store_true",
                   help="Train only the conditioning embedders")
    g.add_argument("--unfreeze_epoch", type=int, default=None,
                   help="Epoch at which to unfreeze --unfreeze_layers, or the whole backbone")
    g.add_argument("--unfreeze_layers", nargs="+", default=None, metavar="LAYER",
                   help="embedding, rnn_1 ... rnn_N, linear")

    args = parse_with_config(p, argv)
    check_training_args(p, args)
    finetune_conditional(args)


if __name__ == "__main__":
    setup_logging()
    main()
