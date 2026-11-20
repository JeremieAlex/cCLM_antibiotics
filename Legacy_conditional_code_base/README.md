# Legacy cCLM

The code used for the results in the paper, kept as it was so the published
molecules can be reproduced. For new work use the `cclm` package in the repository
root.

| File | |
|---|---|
| `model_cCLM.py` | 3-layer LSTM (512 units) with a 6-layer MLP property embedder |
| `helper_functions.py` | vocabulary, dataset, scaling, canonicalization, early stopping |
| `train_cCLM.py` | training: scale properties, canonicalize, build the vocabulary, train |
| `sample_final_cCLM.py` | sampling from a trained checkpoint |
| `analysis.ipynb` | analysis of generated molecules used for the paper figures |

The property vector is scaled with a RobustScaler, embedded to 256 dimensions and
concatenated to the 256-dimensional token embedding at every step. The scaler is
saved as a separate `.pkl` file and is needed for sampling.

Run the scripts from inside this directory. Dependencies are listed in
`../requirements.txt`.

## Train

```bash
python train_cCLM.py \
    --smiles_file input.csv \
    --output_csv_scaled scaled.csv \
    --output_csv_scaled_cleaned scaled_cleaned.csv \
    --scaler_file scaler.pkl \
    --model_file model_out \
    --voc_file Voc \
    --property_columns binary_activity \
    --property_range 0.5 1.0 \
    --batch_size 64 \
    --learning_rate 0.0001
```

`--property_columns` takes one or more CSV columns. `--property_range` takes a
LOW HIGH pair per property; it sets the targets of the molecules sampled at the
start of each epoch as a check (written to `epoch_samples_PT_val/`). Molecules are
kept if they have 11 to 99 heavy atoms made of C, N, O, F, S, Cl and Br. Histograms
of the scaled properties are written to `property_check.pdf`.

A checkpoint `<model_file>_epoch{E}_step0.ckpt` is saved at the start of every
epoch after the first, and the losses go to `<model_file>_loss_log.csv`. Training
stops when the validation loss has not improved by 0.5 for 2 epochs, or after 50
epochs. The model at that point is saved in the current directory as
`Final_model_epoch{E}_step{S}_optimal.ckpt`; this is the model used in the paper.

## Sample

```bash
python sample_final_cCLM.py \
    --n 500 \
    --model_path Final_model_epoch<E>_step<S>_optimal.ckpt \
    --output_file sampled.csv \
    --scaler_file scaler.pkl \
    --voc_file Voc \
    --property_columns binary_activity \
    --prop 1.0 \
    --batch_size 50
```

`--prop` gives one target value per property. With
`--use_range --property_ranges LOW HIGH ...` each batch draws a random target from
the given range instead. The output has the columns `smiles, agent_likelihood,
property_seed`. Sequences are limited to 150 tokens. `agent_likelihood` also counts
tokens drawn after a molecule has ended, so its value depends on the batch size.

## Analysis

`analysis.ipynb` has one cell per analysis; set the `FILE_PATH_HERE` entries at the
top of a cell and run it.

1. Canonicalize a generated set and draw a random sample of 50,000 valid molecules.
2. Coverage of each generated molecule by a reference set (the fraction of its
   Morgan environments found in the reference), with filtered files written at
   coverage thresholds of 1.0 and 0.9.
3. Diversity of the hits in a set: distinct Morgan environments and LeaderPicker
   clusters.
4. UMAP of Morgan fingerprints for generated hits and known hits.
5. UMAP of 15 physicochemical properties for generated hits and training hits.
6. Wasserstein distance per property between generated libraries and a reference set.

Hits are molecules with a predicted `binary_activity` above 0.5. Cells 4 and 5 need
`umap-learn`; cells 5 and 6 use the SA and NP scores from the RDKit Contrib directory.
