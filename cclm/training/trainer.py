"""Training loop shared by pretraining and finetuning."""

import json
import logging
import platform
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import rdkit
import torch
from rdkit import RDLogger
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from cclm.training.cond_dropout import build_conditioning_masks
from cclm.training.dataset import ConditionalDataset, split_train_val, vocab_checker
from cclm.training.early_stopping import EarlyStopping
from cclm.training.preprocessing import (
    describe_specs,
    parse_property_bins,
    parse_property_ranges,
    sample_property_conditions,
)
from cclm.utils.chemistry import augment_smiles, parse_generated_smiles

logger = logging.getLogger(__name__)


def prepare_datasets(args, model, smiles, properties, scaffold_fps):
    """Split by molecule, then apply SMILES augmentation to the training part only."""
    (smiles, properties, scaffold_fps), val = split_train_val(smiles, properties, scaffold_fps, args.val_split)
    in_vocab = vocab_checker(model.vocabulary, model.tokenizer)

    if args.augment_multiplier and args.augment_multiplier > 1:
        smiles, properties, scaffold_fps = augment_smiles(
            smiles, args.augment_multiplier, properties, scaffold_fps, is_valid=in_vocab
        )

    if args.augment_actives_multiplier:
        cc = model.conditioning_config
        if properties is None or cc.num_binary_properties == 0:
            raise ValueError("--augment_actives_multiplier needs --binary_property_columns")
        # The first binary column marks actives.
        actives = np.flatnonzero(properties[:, cc.num_scalar_properties] == 1)
        m = args.augment_actives_multiplier + 1
        aug_smiles, aug_props, aug_fps = augment_smiles(
            [smiles[i] for i in actives], m, properties[actives],
            scaffold_fps[actives] if scaffold_fps is not None else None, is_valid=in_vocab,
        )
        extra = [i for i in range(len(aug_smiles)) if i % m]  # drop the canonical copies
        smiles = smiles + [aug_smiles[i] for i in extra]
        properties = np.concatenate([properties, aug_props[extra]])
        if scaffold_fps is not None:
            scaffold_fps = np.concatenate([scaffold_fps, aug_fps[extra]])
        logger.info(f"Added {len(extra):,} random SMILES for {len(actives):,} actives")

    kw = dict(vocabulary=model.vocabulary, tokenizer=model.tokenizer, randomize=args.randomize_smiles)
    train = ConditionalDataset(smiles, properties=properties, scaffold_fps=scaffold_fps, **kw)
    valid = ConditionalDataset(val[0], properties=val[1], scaffold_fps=val[2], **kw)
    logger.info(f"Train / validation: {len(train):,} / {len(valid):,}")
    return train, valid


def sample_specs(args, model):
    """Property targets for the per-epoch sampling check, or None to sample unconditionally."""
    cc = model.conditioning_config
    if not cc.has_property_conditioning or not (args.property_ranges or args.sample_property_bins):
        return None
    n_flat = cc.num_scalar_properties + cc.num_binary_properties
    specs = parse_property_ranges(args.property_ranges, n_flat) if args.property_ranges else [None] * n_flat
    if args.sample_property_bins:
        specs += parse_property_bins(args.sample_property_bins, cc.num_bins, cc.num_binned_properties)
    else:
        specs += [None] * cc.num_binned_properties
    return specs


def _validity(smiles):
    RDLogger.DisableLog("rdApp.*")
    valid = [s for s in smiles if parse_generated_smiles(s) is not None]
    RDLogger.EnableLog("rdApp.*")
    return 100 * len(valid) / len(smiles), 100 * len(set(valid)) / max(len(valid), 1), valid[:3]


def fit(args, model, train_data, val_data, device, on_epoch_start=None):
    """Train ``model`` and save it to ``args.output_model`` after every epoch.

    The checkpoint with the lowest validation NLL (the early-stopping metric) is
    also kept as ``<output_model>.best.ckpt``.
    ``on_epoch_start(epoch)`` may change which parameters are trainable; if it
    returns True the optimizer and scheduler are rebuilt.
    Writes ``<output_model>.loss_log.csv`` and ``<output_model>.run_summary.json``.
    """
    start = time.time()
    loader_kw = dict(batch_size=args.batch_size, collate_fn=ConditionalDataset.collate,
                     num_workers=args.num_workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_data, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kw)

    def make_optimizer():
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
        return opt, torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)

    optimizer, scheduler = make_optimizer()
    stopper = EarlyStopping(args.patience, args.es_threshold)
    specs = sample_specs(args, model)
    cc = model.conditioning_config
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{n_params:,} parameters, {len(model.vocabulary)} tokens, device {device}")
    if specs:
        logger.info(f"Per-epoch sampling check: {describe_specs(model.property_names, specs)}")

    log = []
    best_metric, best_epoch = float("inf"), None
    for epoch in range(1, args.num_epochs + 1):
        if on_epoch_start is not None and on_epoch_start(epoch):
            optimizer, scheduler = make_optimizer()
        epoch_start = time.time()

        model.set_mode("training")
        train_nlls = []
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}", leave=False)
        for seqs, props, fps in bar:
            seqs = seqs.to(device)
            props = props.to(device) if props is not None else None
            fps = fps.to(device) if fps is not None else None
            prop_mask, scaf_mask = build_conditioning_masks(
                seqs.size(0), props.size(1) if props is not None else 0, args.conditioning_dropout,
                property_dropout=args.property_dropout, has_scaffold=fps is not None,
                mode=args.conditioning_dropout_mode, device=device,
            )
            loss = model.likelihood(seqs, props, fps, prop_mask, scaf_mask).mean()
            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            train_nlls.append(loss.item())
            bar.set_postfix(nll=f"{np.mean(train_nlls[-50:]):.3f}")
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Validation NLL with the true conditioning and with all conditioning replaced by the null.
        model.set_mode("inference")
        val_nlls, uncond_nlls = [], []
        with torch.no_grad():
            for seqs, props, fps in val_loader:
                seqs = seqs.to(device)
                props = props.to(device) if props is not None else None
                fps = fps.to(device) if fps is not None else None
                val_nlls.append(model.likelihood(seqs, props, fps).mean().item())
                if (props is not None or fps is not None) and not args.no_uncond_val:
                    uncond_nlls.append(model.likelihood(seqs).mean().item())
        val_nll = float(np.mean(val_nlls))
        uncond_nll = float(np.mean(uncond_nlls)) if uncond_nlls else val_nll

        row = {"epoch": epoch, "train_nll": float(np.mean(train_nlls)), "val_nll": val_nll,
               "val_nll_uncond": uncond_nll, "lr": lr, "epoch_time_s": round(time.time() - epoch_start, 1)}
        msg = f"Epoch {epoch}: train {row['train_nll']:.3f}  val {val_nll:.3f}  uncond {uncond_nll:.3f}  lr {lr:.1e}"

        if args.sample_every and epoch % args.sample_every == 0:
            props, mask = None, None
            if specs:
                _, props, mask = sample_property_conditions(
                    args.sample_size, specs, model.scaler, model.property_names,
                    cc.num_scalar_properties, device=device,
                )
            _, smiles, _ = model.sample(args.sample_size, properties=props, property_mask=mask)
            row["valid_pct"], row["unique_pct"], examples = _validity(smiles)
            msg += f"  valid {row['valid_pct']:.1f}%  unique {row['unique_pct']:.1f}%"
            if examples:
                msg += "\n  e.g. " + " | ".join(examples)
        logger.info(msg)
        log.append(row)

        model.save(args.output_model)
        if args.save_every and epoch % args.save_every == 0:
            model.save(f"{args.output_model}.epoch{epoch}.ckpt")
        metric = uncond_nll if args.early_stopping_metric == "unconditional" else val_nll
        if metric < best_metric:
            best_metric, best_epoch = metric, epoch
            model.save(f"{args.output_model}.best.ckpt")

        stopper(metric, epoch)
        if stopper.stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    pd.DataFrame(log).to_csv(f"{args.output_model}.loss_log.csv", index=False)
    summary = {
        "args": {k: v for k, v in vars(args).items()},
        "conditioning": cc.as_dict(),
        "property_names": model.property_names,
        "n_params": n_params,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "epochs_completed": len(log),
        "early_stopped": stopper.stop,
        "best_epoch": best_epoch,
        "final": log[-1] if log else None,
        "total_time_min": round((time.time() - start) / 60, 2),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "rdkit": rdkit.__version__,
        "platform": platform.platform(),
        "device": str(device),
    }
    with open(f"{args.output_model}.run_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved {args.output_model}")
