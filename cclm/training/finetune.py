"""Finetune a pretrained conditional CLM. Architecture, vocabulary, scaler and bin edges come from the checkpoint."""

import logging
import os

from cclm.model.model import Model
from cclm.training.dataset import vocab_checker
from cclm.training.preprocessing import load_and_preprocess_csv, load_scaler
from cclm.training.trainer import fit, prepare_datasets
from cclm.utils.device import get_device, set_seed
from cclm.utils.scaffold import resolve_scaffold_fp

logger = logging.getLogger(__name__)

BACKBONE = ("_embedding", "_rnn", "_linear")

# Names accepted by --unfreeze_layers.
LAYERS = {
    "embedding": lambda n: n.startswith("_embedding."),
    "linear": lambda n: n.startswith("_linear."),
}


def _layer_matcher(name: str, num_layers: int):
    if name in LAYERS:
        return LAYERS[name]
    if name.startswith("rnn_") and name[4:].isdigit() and 1 <= int(name[4:]) <= num_layers:
        suffix = f"_l{int(name[4:]) - 1}"
        return lambda n: n.startswith("_rnn.") and n.split(".")[-1].endswith(suffix)
    valid = list(LAYERS) + [f"rnn_{i}" for i in range(1, num_layers + 1)]
    raise ValueError(f"Unknown layer '{name}'. Valid: {valid}")


def _set_trainable(model: Model, matchers, trainable: bool) -> None:
    for name, p in model.network.named_parameters():
        if any(m(name) for m in matchers):
            p.requires_grad = trainable
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{n:,} trainable parameters")


def finetune_conditional(args) -> None:
    set_seed(args.seed)
    device = get_device(args.device)

    model = Model.load(args.pretrained_model, mode="training", device=device)
    cc = model.conditioning_config
    logger.info(f"Loaded {args.pretrained_model}: {cc.num_properties} properties, scaffold fp {cc.scaffold_fp_size}")

    scaffold_fp_type = scaffold_fp_size = None
    if cc.has_scaffold_conditioning:
        if not args.scaffold_column:
            raise ValueError("The pretrained model is scaffold-conditioned; pass --scaffold_column")
        scaffold_fp_type, scaffold_fp_size = resolve_scaffold_fp(cc, args.scaffold_fp_type, args.scaffold_fp_size)
    elif args.scaffold_column:
        raise ValueError("The pretrained model has no scaffold conditioning")

    computed = list(args.compute_properties or [])
    requested = (computed + [c for c in args.property_columns or [] if c not in computed]
                 + list(args.binary_property_columns or []) + list(args.bin_property_columns or []))
    if model.property_names and requested != model.property_names:
        raise ValueError(
            f"Property columns {requested} differ from the checkpoint's {model.property_names}; "
            "names and order must match"
        )

    scaler = load_scaler(args.premade_scaler) or model.scaler
    data = load_and_preprocess_csv(
        args.smiles_file,
        smiles_column=args.smiles_column,
        property_columns=args.property_columns,
        binary_property_columns=args.binary_property_columns,
        bin_property_columns=args.bin_property_columns,
        num_bins=cc.num_bins or args.num_bins,
        bin_strategy=args.bin_strategy,
        premade_bin_edges=model.bin_edges,
        compute_properties=args.compute_properties,
        scaffold_column=args.scaffold_column,
        scaffold_fp_type=scaffold_fp_type,
        scaffold_fp_bits=scaffold_fp_size,
        scaler_path=args.scaler_file,
        scaler=scaler,
    )

    layout = (data["num_scalar_properties"], data["num_binary_properties"], data["num_binned_properties"])
    expected = (cc.num_scalar_properties, cc.num_binary_properties, cc.num_binned_properties)
    if layout != expected:
        raise ValueError(
            f"The data has {layout[0]} scalar / {layout[1]} binary / {layout[2]} binned properties, "
            f"the checkpoint expects {expected[0]} / {expected[1]} / {expected[2]}"
        )
    if model.property_names and model.property_names != data["property_names"]:
        raise ValueError(
            f"Property columns {data['property_names']} differ from the checkpoint's {model.property_names}; "
            "names and order must match"
        )
    model.property_names = data["property_names"]
    model.scaler = data["scaler"]
    model.bin_edges = {**model.bin_edges, **data["bin_edges"]}
    if data["bin_occupancy"]:
        model.bin_occupancy = {**model.bin_occupancy, **data["bin_occupancy"]}

    # Drop molecules with tokens the pretrained vocabulary does not have.
    in_vocab = vocab_checker(model.vocabulary, model.tokenizer)
    keep = [i for i, s in enumerate(data["smiles"]) if in_vocab(s)]
    if len(keep) < len(data["smiles"]):
        logger.warning(f"Dropped {len(data['smiles']) - len(keep):,} molecules with unknown tokens")
    if not keep:
        raise ValueError("No molecules left after removing unknown tokens")
    smiles = [data["smiles"][i] for i in keep]
    props = data["properties"][keep] if data["properties"] is not None else None
    fps = data["scaffold_fps"][keep] if data["scaffold_fps"] is not None else None

    num_layers = model.network.get_params()["num_layers"]
    unfreeze = [_layer_matcher(n, num_layers) for n in (args.unfreeze_layers or [])]
    backbone = [lambda n, p=p: n.startswith(p + ".") for p in BACKBONE]
    if args.unfreeze_epoch and not args.freeze_backbone:
        raise ValueError("--unfreeze_epoch requires --freeze_backbone")
    if args.freeze_backbone:
        _set_trainable(model, backbone, False)
        if unfreeze and not args.unfreeze_epoch:
            _set_trainable(model, unfreeze, True)

    def on_epoch_start(epoch):
        if args.freeze_backbone and args.unfreeze_epoch == epoch:
            logger.info(f"Epoch {epoch}: unfreezing {args.unfreeze_layers or 'the backbone'}")
            _set_trainable(model, unfreeze or backbone, True)
            return True
        return False

    os.makedirs(os.path.dirname(args.output_model) or ".", exist_ok=True)
    train, val = prepare_datasets(args, model, smiles, props, fps)
    fit(args, model, train, val, device, on_epoch_start=on_epoch_start)
