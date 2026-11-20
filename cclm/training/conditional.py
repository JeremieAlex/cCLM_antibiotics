"""Train a conditional CLM from scratch."""

import logging
import os

from cclm.model.conditioning import ConditioningConfig
from cclm.model.model import Model
from cclm.training.preprocessing import load_and_preprocess_csv, load_scaler
from cclm.training.trainer import fit, prepare_datasets
from cclm.utils.device import get_device, set_seed
from cclm.utils.scaffold import DEFAULT_SCAFFOLD_FP_TYPE

logger = logging.getLogger(__name__)


def train_conditional(args) -> None:
    set_seed(args.seed)
    device = get_device(args.device)

    blank = args.blank_conditioning or args.blank_binary_conditioning
    if blank and (args.property_columns or args.compute_properties
                  or args.binary_property_columns or args.bin_property_columns):
        raise ValueError("--blank_conditioning cannot be combined with property columns")

    data = load_and_preprocess_csv(
        args.smiles_file,
        smiles_column=args.smiles_column,
        property_columns=args.property_columns,
        binary_property_columns=args.binary_property_columns,
        bin_property_columns=args.bin_property_columns,
        num_bins=args.num_bins,
        bin_strategy=args.bin_strategy,
        compute_properties=args.compute_properties,
        scaffold_column=args.scaffold_column,
        scaffold_fp_type=args.scaffold_fp_type,
        scaffold_fp_bits=args.scaffold_fp_size,
        scaler_path=args.scaler_file,
        scaler=load_scaler(args.premade_scaler),
    )

    if blank:
        # Reserve property slots that stay empty (null embedding) until finetuning.
        n_scalar, n_binary, n_binned = args.blank_conditioning or 0, args.blank_binary_conditioning or 0, 0
        args.conditioning_dropout = 1.0
    else:
        n_scalar = data["num_scalar_properties"]
        n_binary = data["num_binary_properties"]
        n_binned = data["num_binned_properties"]

    fp_size = data["scaffold_fps"].shape[1] if data["scaffold_fps"] is not None else 0
    cc = ConditioningConfig(
        num_scalar_properties=n_scalar,
        num_binary_properties=n_binary,
        num_binned_properties=n_binned,
        num_bins=data["num_bins"],
        scaffold_fp_size=fp_size,
        scaffold_fp_type=(args.scaffold_fp_type or DEFAULT_SCAFFOLD_FP_TYPE) if fp_size else None,
        condition_embedding_size=args.condition_embedding_size,
        property_embedder_layers=args.property_embedder_layers,
        binary_embedder_layers=args.binary_embedder_layers,
        binned_embedder_layers=args.binned_embedder_layers,
        scaffold_embedder_layers=args.scaffold_embedder_layers,
        conditioning_dropout=args.conditioning_dropout,
        use_learned_null=not args.no_learned_null,
    )

    embedding_size = args.embedding_size
    if args.rnn_input_size is not None:
        embedding_size = args.rnn_input_size - cc.total_conditioning_size
        if embedding_size <= 0:
            raise ValueError(f"--rnn_input_size must be larger than the conditioning size ({cc.total_conditioning_size})")

    model = Model(
        vocabulary=data["vocabulary"],
        tokenizer=data["tokenizer"],
        network_params=dict(layer_size=args.layer_size, num_layers=args.num_layers, cell_type=args.cell_type,
                            embedding_layer_size=embedding_size, dropout=args.dropout),
        conditioning_config=cc,
        max_sequence_length=args.max_sequence_length,
        device=device,
        scaler=data["scaler"],
        bin_edges=data["bin_edges"],
        bin_occupancy=data["bin_occupancy"],
        property_names=[] if blank else data["property_names"],
    )
    if model.property_names:
        logger.info(f"Conditioning on {model.property_names}")

    os.makedirs(os.path.dirname(args.output_model) or ".", exist_ok=True)
    train, val = prepare_datasets(args, model, data["smiles"], data["properties"], data["scaffold_fps"])
    fit(args, model, train, val, device)
