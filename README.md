# cCLM: conditional chemical language model

An RNN SMILES generator conditioned on molecular properties (computed
descriptors, your own continuous columns, binary labels or binned scores) and/or
a Murcko scaffold.

- `cclm/`, `scripts/`, `configs/`: the package.
- `Legacy_conditional_code_base/`: the original code used for the paper, kept
  unchanged for reproducing the published results (see its own README).

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

Tested with Python 3.11, PyTorch 2.4.1 (CUDA 12.4), RDKit 2024.03.5.

## Data

A CSV with a `SMILES` column (or a `.smi` file). Property columns are optional;
descriptors can be computed from the SMILES instead. Molecules are canonicalized
and kept if they have 11-99 heavy atoms made of C, N, O, F, S, Cl, Br and I.
Rows with missing property values are dropped.

The configs use `data/October2024_cleaned_chembl_smiles_USA300_preds_LogP_TPSA_MW.csv`
(ChEMBL with LogP, TPSA, MW and a predicted USA300 activity score), available from
<!-- TODO: Zenodo DOI -->.

## Train

```bash
python scripts/train_conditional.py --config configs/train_physchem.toml
```

Any config value can be overridden on the command line (`--num_epochs 10`), or
everything can be given on the command line:

```bash
python scripts/train_conditional.py --smiles_file data/mols.csv --output_model models/model.ckpt \
    --compute_properties LogP MW QED TPSA --num_epochs 30
```

Property options:

| Option | Meaning |
|---|---|
| `--compute_properties LogP MW ...` | RDKit descriptors computed from the SMILES (list them with `python -c "from cclm.training.property_calculator import print_available; print_available()"`) |
| `--property_columns a b` | continuous CSV columns, scaled with a RobustScaler |
| `--binary_property_columns y` | 0/1 CSV columns |
| `--bin_property_columns s --num_bins 10` | a continuous column turned into bins (equal-count by default, `--bin_strategy uniform` for equal width). With equal-count bins the top bin is the top 1/num_bins of the data; use more bins or uniform bins to target only the highest values |
| `--scaffold_column auto` | condition on the Murcko scaffold (or give a column of scaffold SMILES) |

The order of the property vector is computed descriptors, then `--property_columns`,
then binary, then binned columns. The scaler, bin edges and column names are saved
in the checkpoint, and sampling checks the columns you pass against them.

The checkpoint is written after every epoch, so it ends as the last epoch trained.
The epoch with the lowest validation NLL is also kept as `<model>.best.ckpt`.
Training also writes `<model>.loss_log.csv` and `<model>.run_summary.json`. The loss log has two validation curves: `val_nll` with
the true conditioning and `val_nll_uncond` with all conditioning hidden.

**Conditioning dropout.** `--conditioning_dropout` (default 0.1) is the chance a
molecule is shown none of its conditioning, which trains a learned "null" embedding.
`--property_dropout` (defaults to the same value) hides single columns, so the model
can be given only some properties at sampling time.

## Check the conditioning

```bash
python scripts/validate_conditioning.py --config configs/validate_physchem.toml
```

This sweeps one property through a list of targets with the others held fixed,
samples at each target, recomputes the properties with RDKit and prints the Pearson
r between target and achieved mean, the per-molecule MAE, the mean bias, the MAE of
the held properties, validity and uniqueness. Add `full_molecule` or
`scaffold_smiles` to hold a scaffold fixed and also report scaffold retention
(`configs/validate_scaffold.toml`).

## Sample

```bash
python scripts/sample.py --model models/physchem.ckpt --mode property \
    --property_columns LogP TPSA MW --property_ranges 2 4 50 110 300 450 \
    --n 1000 --output results/sample.csv
```

Ranges are `LOW HIGH` per property in training order and original units. Each
molecule gets its own target drawn uniformly from the ranges. Write `none` (or a
quoted `'*'`) instead of a pair to leave a property unspecified:

```bash
--property_ranges 2 4 none 300 450
```

Binned columns take `--property_bins` with a bin index (0-based), an inclusive range
such as `7-9`, `top`, or `none`. Other modes:

```bash
python scripts/sample.py --model models/scaffold.ckpt --mode scaffold \
    --full_molecule "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O" --n 500 --output results/scaffold.csv
python scripts/sample.py --model models/physchem.ckpt --mode vanilla --n 500 --output results/vanilla.csv
```

`property_scaffold` combines both. The output has unique canonical SMILES, their
NLL (sorted, lowest first; computed without temperature so it is comparable across
temperatures), the target each molecule was sampled with, and the scaffold.

## Pretrain and finetune

Pretrain on a large unlabeled set with empty property slots, then finetune on the
labeled set:

```bash
python scripts/train_conditional.py --config configs/pretrain_blank.toml
python scripts/finetune_conditional.py --config configs/finetune.toml
```

A model pretrained with real properties can also be finetuned, as long as the
finetuning run uses the same property columns in the same order. The architecture,
vocabulary, scaler and bin edges come from the checkpoint, and molecules with tokens
outside its vocabulary are dropped. `--freeze_backbone` trains only the conditioning
layers; add `--unfreeze_epoch N` to unfreeze the backbone (or the layers named in
`--unfreeze_layers embedding rnn_1 rnn_2 rnn_3 linear`) at epoch N.

## Configs

| Config | |
|---|---|
| `train_physchem.toml` | LogP, TPSA, MW |
| `validate_physchem.toml`, `sample_physchem.toml` | sweep MW / sample a property window |
| `train_scaffold.toml` | LogP, TPSA, MW + Murcko scaffold |
| `validate_scaffold.toml`, `sample_scaffold.toml` | the same with the ciprofloxacin scaffold |
| `train_activity.toml` | predicted USA300 activity with the published settings (3x512 LSTM, no conditioning dropout) |
| `sample_activity.toml` | sample at high predicted activity |
| `pretrain_blank.toml`, `finetune.toml` | two-phase training |

The activity model is trained without conditioning dropout, so it should not be
sampled with `none` ranges or in vanilla mode.

## How it works

SMILES are tokenized with the REINVENT regex tokenizer. Each property type has its
own head: scalar and binary columns go through MLPs, binned columns through an
embedding table, and the scaffold fingerprint through another MLP. When several
heads are used, their outputs are concatenated and projected back to the embedding
size. The property embedding (and the scaffold embedding) is concatenated to the
token embedding at every step of a GRU or LSTM. Hidden values are replaced by an
out-of-range constant (scalar and binary) or an "unspecified" row (binned), and a
molecule with nothing visible gets the learned null embedding. Training minimizes
the teacher-forced NLL.

## Tests

```bash
pip install pytest
pytest tests
```

The tests train tiny models on `tests/data/small.csv` on the CPU and exercise every
training, finetuning, sampling and validation option (about 2 minutes).
