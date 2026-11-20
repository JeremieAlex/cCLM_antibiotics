#!/usr/bin/env python
"""Train a property- and/or scaffold-conditioned chemical language model.

Example:
    python scripts/train_conditional.py --smiles_file data/molecules.csv \\
        --output_model models/model.ckpt --compute_properties LogP MW QED TPSA

List the descriptors that can be computed:
    python -c "from cclm.training.property_calculator import print_available; print_available()"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cclm.threads import limit_threads  # noqa: E402

limit_threads()

from cclm.cli import add_training_args, check_training_args, setup_logging  # noqa: E402
from cclm.training.conditional import train_conditional  # noqa: E402
from cclm.utils.config import parse_with_config  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_args(p)
    p.add_argument("--scaffold_fp_size", type=int, default=2048)
    p.add_argument("--blank_conditioning", type=int, default=None, metavar="N",
                   help="Reserve N empty scalar property slots, to be filled by finetuning")
    p.add_argument("--blank_binary_conditioning", type=int, default=None, metavar="N",
                   help="Reserve N empty binary property slots")
    p.add_argument("--no_learned_null", action="store_true",
                   help="Use zeros instead of a learned embedding for hidden conditioning")

    g = p.add_argument_group("model")
    g.add_argument("--cell_type", default="gru", choices=["gru", "lstm"])
    g.add_argument("--layer_size", type=int, default=512)
    g.add_argument("--num_layers", type=int, default=3)
    g.add_argument("--embedding_size", type=int, default=256, help="Token embedding size")
    g.add_argument("--rnn_input_size", type=int, default=None,
                   help="Fix the total RNN input size; the token embedding gets what conditioning leaves")
    g.add_argument("--dropout", type=float, default=0.0)
    g.add_argument("--max_sequence_length", type=int, default=256)
    g.add_argument("--condition_embedding_size", type=int, default=256)
    g.add_argument("--property_embedder_layers", type=int, default=4)
    g.add_argument("--binary_embedder_layers", type=int, default=2)
    g.add_argument("--binned_embedder_layers", type=int, default=2)
    g.add_argument("--scaffold_embedder_layers", type=int, default=3)

    args = parse_with_config(p, argv)
    check_training_args(p, args)
    train_conditional(args)


if __name__ == "__main__":
    setup_logging()
    main()
