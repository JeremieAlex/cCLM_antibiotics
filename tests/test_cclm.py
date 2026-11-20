import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import ROOT, train
from cclm.model import ConditioningConfig, Model, SMILESTokenizer, create_vocabulary
from cclm.training.cond_dropout import build_conditioning_masks
from cclm.training.preprocessing import (
    assign_bins,
    compute_bin_edges,
    load_and_preprocess_csv,
    parse_property_bins,
    parse_property_ranges,
    sample_property_conditions,
)

CIPRO = "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"


# ---------- units ----------

def test_tokenizer_roundtrip():
    tok = SMILESTokenizer()
    for smi in ["CC(=O)Oc1ccccc1C(=O)O", "Clc1ccc(Br)cc1", "C[C@@H](N)C(=O)O", "C1CC%10CC1%10", "[NH4+]"]:
        tokens = tok.tokenize(smi)
        assert tok.untokenize(tokens) == smi
    assert "Cl" in tok.tokenize("Clc1ccccc1") and "[C@@H]" in tok.tokenize("C[C@@H](N)O")


def test_vocabulary_dictionary_roundtrip():
    tok = SMILESTokenizer()
    vocab = create_vocabulary(["CCO", "c1ccccc1Cl", "C[NH3+]"], tok)
    from cclm.model import Vocabulary
    again = Vocabulary.load_from_dictionary(vocab.get_dictionary())
    assert again == vocab
    assert vocab["$"] == 0 and vocab["^"] == 1
    with pytest.raises(KeyError):
        vocab.encode(["Xx"])


def test_bins():
    vals = np.arange(100, dtype=float)
    edges = compute_bin_edges(vals, 4)
    idx = assign_bins(vals, edges)
    assert np.bincount(idx.astype(int)).tolist() == [25, 25, 25, 25]
    assert assign_bins(np.array([-1e9, 1e9]), edges).tolist() == [0, 3]
    with pytest.raises(ValueError):
        assign_bins(np.array([np.nan]), edges)
    uni = compute_bin_edges(vals, 3, "uniform")
    assert np.allclose(uni, [0, 33, 66, 99])


def test_spec_parsing():
    assert parse_property_ranges(["1", "2", "none", "*", "3", "4"], 4) == [(1, 2), None, None, (3, 4)]
    assert parse_property_ranges([0.5, 1.0], 1) == [(0.5, 1.0)]
    with pytest.raises(ValueError):
        parse_property_ranges(["1", "2", "3"], 2)
    with pytest.raises(ValueError):
        parse_property_ranges(["1", "2"], 2)
    assert parse_property_bins(["top", "2-3", "none", "0"], 5, 4) == [("bin", 4, 4), ("bin", 2, 3), None, ("bin", 0, 0)]
    for bad in (["5"], ["3-1"], ["x"]):
        with pytest.raises(ValueError):
            parse_property_bins(bad, 5, 1)


def test_masks():
    torch.manual_seed(0)
    p, s = build_conditioning_masks(10000, 3, 0.0, property_dropout=0.3, has_scaffold=True)
    assert p.shape == (10000, 3) and s.shape == (10000, 1)
    assert abs(1 - p.mean().item() - 0.3) < 0.02 and abs(1 - s.mean().item() - 0.3) < 0.02
    p, _ = build_conditioning_masks(10000, 3, 0.2, property_dropout=0.0)
    rows = p.sum(1)
    assert set(rows.unique().tolist()) == {0.0, 3.0} and abs((rows == 0).float().mean().item() - 0.2) < 0.02
    p, _ = build_conditioning_masks(64, 2, 0.5, mode="batch")
    assert p.unique().numel() == 1
    p, _ = build_conditioning_masks(10000, 3, 0.1, property_dropout=0.5, mode="example")
    assert set(p.sum(1).unique().tolist()) == {0.0, 3.0}


def test_sample_property_conditions_scaling():
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler().fit(pd.DataFrame({"a": [0.0, 1, 2, 3, 4], "b": [10.0, 20, 30, 40, 50]}))
    raw, t, mask = sample_property_conditions(8, [(2, 2), None, ("bin", 3, 3)], scaler, ["a", "b", "c"], 2)
    assert np.allclose(raw[:, 0], 2) and np.isnan(raw[:, 1]).all() and np.allclose(raw[:, 2], 3)
    assert torch.allclose(t[:, 0], torch.zeros(8))  # 2 is the median of column a
    assert mask[:, 1].sum() == 0 and mask[:, 0].sum() == 8
    assert torch.allclose(t[:, 2], torch.full((8,), 3.0))


def test_preprocessing_drops_bad_rows(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({
        "SMILES": ["CCCCCCCCCCCO", "c1ccccc1CCCCCCN", "not_a_smiles", "CCCCCCCCCCCCCCN", "C[Si](C)(C)CCCCCCCCCC"],
        "y": [1.0, np.nan, 2.0, 3.0, 4.0],
    }).to_csv(csv, index=False)
    d = load_and_preprocess_csv(str(csv), property_columns=["y"])
    assert len(d["smiles"]) == 2 and not np.isnan(d["properties"]).any()


def test_smi_input(tmp_path, data_csv):
    smi = tmp_path / "mols.smi"
    smiles = pd.read_csv(data_csv).SMILES.head(200)
    smi.write_text("# comment\n" + "\n".join(f"{s}\tid{i}" for i, s in enumerate(smiles)) + "\n")
    d = load_and_preprocess_csv(str(smi), compute_properties=["MW", "LogP"])
    assert len(d["smiles"]) > 150 and d["properties"].shape[1] == 2


# ---------- model behaviour ----------

def _tiny_model(cc):
    vocab = create_vocabulary(["CCO", "c1ccccc1N", "CC(=O)Cl"], SMILESTokenizer())
    torch.manual_seed(0)
    m = Model(vocab, SMILESTokenizer(), dict(layer_size=32, embedding_layer_size=16), cc, max_sequence_length=40,
              mode="inference")
    for p in m.parameters():  # make the null embeddings non-trivial
        torch.nn.init.normal_(p, std=0.3)
    return m


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_sampled_nll_matches_likelihood(cell):
    cc = ConditioningConfig(num_scalar_properties=2, num_binary_properties=1, num_binned_properties=1,
                            num_bins=4, scaffold_fp_size=8, condition_embedding_size=16)
    vocab = create_vocabulary(["CCO", "c1ccccc1N"], SMILESTokenizer())
    torch.manual_seed(0)
    m = Model(vocab, SMILESTokenizer(), dict(layer_size=32, embedding_layer_size=16, cell_type=cell), cc,
              max_sequence_length=30, mode="inference")
    props = torch.tensor([[0.5, -1.0, 1.0, 2.0]]).repeat(16, 1)
    fps = torch.rand(16, 8).round()
    mask = torch.ones(16, 4)
    mask[::2, 1] = 0
    for temperature in (1.0, 0.5, 2.0):
        seqs, _, nlls = m.sample(16, props, fps, temperature=temperature, property_mask=mask)
        # Only score up to and including the first STOP token.
        stop = vocab["$"]
        ref = []
        for i in range(16):
            row = seqs[i].tolist()
            end = row.index(stop, 1) + 1 if stop in row[1:] else len(row)
            ref.append(m.likelihood(seqs[i:i + 1, :end], props[i:i + 1], fps[i:i + 1], mask[i:i + 1]).item())
        assert np.allclose(nlls.numpy(), ref, atol=1e-4)


def test_fully_masked_equals_null():
    cc = ConditioningConfig(num_scalar_properties=2, num_binary_properties=1, condition_embedding_size=16,
                            scaffold_fp_size=8)
    m = _tiny_model(cc)
    seqs = torch.randint(2, len(m.vocabulary), (4, 10))
    props = torch.randn(4, 3)
    fps = torch.rand(4, 8)
    masked = m.likelihood(seqs, props, fps, torch.zeros(4, 3), torch.zeros(4))
    null = m.likelihood(seqs)
    assert torch.allclose(masked, null, atol=1e-5)
    # Hiding a column must change the output, and so must changing a visible value.
    full = m.likelihood(seqs, props, fps)
    partial = torch.ones(4, 3)
    partial[:, 0] = 0
    assert not torch.allclose(full, m.likelihood(seqs, props, fps, partial))
    assert not torch.allclose(full, m.likelihood(seqs, props + 1, fps))
    # A hidden column is ignored.
    changed = props.clone()
    changed[:, 0] += 5
    assert torch.allclose(m.likelihood(seqs, props, fps, partial), m.likelihood(seqs, changed, fps, partial))


def test_binned_unspecified_row():
    cc = ConditioningConfig(num_binned_properties=2, num_bins=3, condition_embedding_size=16)
    m = _tiny_model(cc)
    seqs = torch.randint(2, len(m.vocabulary), (3, 8))
    a = m.likelihood(seqs, torch.tensor([[0.0, 2.0]] * 3), property_mask=torch.tensor([[1.0, 0.0]] * 3))
    b = m.likelihood(seqs, torch.tensor([[0.0, 1.0]] * 3), property_mask=torch.tensor([[1.0, 0.0]] * 3))
    c = m.likelihood(seqs, torch.tensor([[0.0, 1.0]] * 3))
    assert torch.allclose(a, b) and not torch.allclose(a, c)


def test_save_load_roundtrip(tmp_path):
    cc = ConditioningConfig(num_scalar_properties=1, num_binned_properties=1, num_bins=3,
                            condition_embedding_size=16, scaffold_fp_size=8, scaffold_fp_type="Morgan")
    m = _tiny_model(cc)
    m.property_names = ["x", "y"]
    m.bin_edges = {"y": [0, 1, 2, 3]}
    path = tmp_path / "m.ckpt"
    m.save(path)
    m2 = Model.load(path)
    seqs = torch.randint(2, len(m.vocabulary), (3, 8))
    props, fps = torch.randn(3, 2), torch.rand(3, 8)
    assert torch.allclose(m.likelihood(seqs, props, fps), m2.likelihood(seqs, props, fps))
    assert m2.property_names == ["x", "y"] and m2.bin_edges == {"y": [0, 1, 2, 3]}
    assert m2.conditioning_config == cc


def test_load_without_learned_null(tmp_path):
    cc = ConditioningConfig(num_scalar_properties=1, condition_embedding_size=16, use_learned_null=False)
    m = _tiny_model(cc)
    m.save(tmp_path / "z.ckpt")
    m2 = Model.load(tmp_path / "z.ckpt")
    assert not m2.conditioning_config.use_learned_null
    seqs = torch.randint(2, len(m.vocabulary), (2, 6))
    assert torch.allclose(m.likelihood(seqs), m2.likelihood(seqs))


# ---------- training, sampling, validation end to end ----------

def _sample(*args):
    import sample
    return sample.main(["--n", "20", "--batch_size", "32", "--device", "cpu", "--seed", "0", *args])


def test_training_outputs(models):
    for path in models.values():
        assert Path(path).exists()
        assert Path(path + ".loss_log.csv").exists()
        assert Path(path + ".best.ckpt").exists()
        summary = json.loads(Path(path + ".run_summary.json").read_text())
        assert summary["epochs_completed"] == 1
        log = pd.read_csv(path + ".loss_log.csv")
        assert np.isfinite(log[["train_nll", "val_nll", "val_nll_uncond"]].values).all()
    m = Model.load(models["mixed"])
    assert m.property_names == ["QED", "LogP", "active", "binary_activity"]
    assert m.conditioning_config.num_bins == 5 and len(m.bin_edges["binary_activity"]) == 6
    assert Model.load(models["blank"]).conditioning_config.num_scalar_properties == 1


def test_sample_modes(models, tmp_path):
    out = str(tmp_path / "s.csv")
    df = _sample("--model", models["physchem"], "--output", out)
    assert len(df) == 20 and df.smiles.is_unique and df.nll.is_monotonic_increasing
    df = _sample("--model", models["physchem"], "--output", out, "--mode", "property",
                 "--property_columns", "LogP", "TPSA", "MW", "--property_ranges", "2", "3", "none", "300", "400")
    seeds = df.property_seed.str.split(",", expand=True)
    assert (seeds[1] == "TPSA=*").all()
    logp = seeds[0].str.split("=").str[1].astype(float)
    assert logp.between(2, 3).all() and logp.nunique() > 1  # one seed per molecule
    df = _sample("--model", models["mixed"], "--output", out, "--mode", "property",
                 "--property_columns", "QED", "LogP", "--binary_property_columns", "active",
                 "--bin_property_columns", "binary_activity", "--property_ranges", "*", "1", "3", "1", "1",
                 "--property_bins", "top")
    assert df.property_seed.str.contains("binary_activity=4.0000").all()
    df = _sample("--model", models["scaffold"], "--output", out, "--mode", "scaffold", "--full_molecule", CIPRO)
    assert df.scaffold.nunique() == 1
    _sample("--model", models["scaffold"], "--output", out, "--mode", "property_scaffold",
            "--property_columns", "LogP", "TPSA", "MW", "--property_ranges", "1", "3", "60", "90", "350", "450",
            "--scaffold_smiles", "c1ccc2ncccc2c1")
    _sample("--model", models["blank"], "--output", out)


def test_sample_errors(models, tmp_path):
    out = str(tmp_path / "s.csv")
    with pytest.raises(ValueError, match="do not match"):
        _sample("--model", models["physchem"], "--output", out, "--mode", "property",
                "--property_columns", "MW", "TPSA", "LogP", "--property_ranges", "1", "2", "1", "2", "1", "2")
    with pytest.raises(ValueError, match="expects"):
        _sample("--model", models["physchem"], "--output", out, "--mode", "property",
                "--property_columns", "LogP", "--property_ranges", "1", "2")
    with pytest.raises(ValueError, match="does not use"):
        _sample("--model", models["scaffold"], "--output", out, "--scaffold_smiles", "c1ccccc1")
    with pytest.raises(ValueError, match="without scaffold"):
        _sample("--model", models["physchem"], "--output", out, "--mode", "scaffold", "--scaffold_smiles", "c1ccccc1")
    with pytest.raises(ValueError, match="does not match"):
        _sample("--model", models["scaffold"], "--output", out, "--mode", "scaffold",
                "--scaffold_smiles", "c1ccccc1", "--scaffold_fp_size", "2048")


def test_validate(models, tmp_path):
    import validate_conditioning
    out = str(tmp_path / "v.csv")
    sdf, r = validate_conditioning.main([
        "--model", models["physchem"], "--property_columns", "LogP", "TPSA", "MW", "--sweep_property", "MW",
        "--sweep_values", "300", "450", "--fixed", "LogP=2.5", "TPSA=70", "--n", "10", "--batch_size", "32",
        "--device", "cpu", "--output", out])
    assert len(sdf) == 2 and Path(out).exists() and (tmp_path / "v_summary.csv").exists()
    sdf, _ = validate_conditioning.main([
        "--model", models["scaffold"], "--property_columns", "LogP", "TPSA", "MW", "--sweep_property", "LogP",
        "--sweep_values", "1", "3", "--fixed", "MW=400", "TPSA=70", "--full_molecule", CIPRO, "--n", "10",
        "--batch_size", "32", "--device", "cpu", "--output", out])
    assert "scaffold_exact_pct" in sdf
    with pytest.raises(ValueError, match="No --fixed"):
        validate_conditioning.main([
            "--model", models["physchem"], "--property_columns", "LogP", "TPSA", "MW", "--sweep_property", "MW",
            "--sweep_values", "300", "--fixed", "LogP=2.5", "--device", "cpu", "--output", out])


def test_finetune_blank(models, tmp_path, data_csv):
    import finetune_conditional
    out = str(tmp_path / "ft.ckpt")
    finetune_conditional.main([
        "--pretrained_model", models["blank"], "--smiles_file", data_csv, "--output_model", out,
        "--property_columns", "binary_activity", "--num_epochs", "2", "--batch_size", "64", "--device", "cpu",
        "--freeze_backbone", "--unfreeze_epoch", "2", "--unfreeze_layers", "linear", "rnn_3",
        "--property_ranges", "0.5", "1.0", "--sample_size", "16"])
    m = Model.load(out)
    pre = Model.load(models["blank"])
    assert m.property_names == ["binary_activity"] and m.scaler is not None
    w = dict(m.network.named_parameters())
    w0 = dict(pre.network.named_parameters())
    assert torch.equal(w["_embedding.weight"], w0["_embedding.weight"])
    assert torch.equal(w["_rnn.weight_ih_l0"], w0["_rnn.weight_ih_l0"])
    assert not torch.equal(w["_rnn.weight_ih_l2"], w0["_rnn.weight_ih_l2"])
    assert not torch.equal(w["_linear.weight"], w0["_linear.weight"])
    _sample("--model", out, "--output", str(tmp_path / "s.csv"), "--mode", "property",
            "--property_columns", "binary_activity", "--property_ranges", "0.8", "1")


def test_finetune_same_columns(models, tmp_path, data_csv):
    import finetune_conditional
    out = str(tmp_path / "ft2.ckpt")
    base = ["--pretrained_model", models["physchem"], "--smiles_file", data_csv, "--output_model", out,
            "--num_epochs", "1", "--batch_size", "64", "--device", "cpu", "--sample_every", "0"]
    finetune_conditional.main(base + ["--property_columns", "LogP", "TPSA", "MW", "--augment_multiplier", "2"])
    assert Model.load(out).scaler.center_.tolist() == Model.load(models["physchem"]).scaler.center_.tolist()
    with pytest.raises(ValueError, match="differ"):
        finetune_conditional.main(base + ["--property_columns", "MW", "TPSA", "LogP"])
    with pytest.raises(ValueError, match="differ"):
        finetune_conditional.main(base + ["--property_columns", "LogP"])
    with pytest.raises(ValueError, match="scaffold"):
        finetune_conditional.main(base + ["--property_columns", "LogP", "TPSA", "MW", "--scaffold_column", "auto"])


def test_finetune_scaffold_and_bins(models, tmp_path, data_csv):
    import finetune_conditional
    out = str(tmp_path / "ft3.ckpt")
    finetune_conditional.main([
        "--pretrained_model", models["scaffold"], "--smiles_file", data_csv, "--output_model", out,
        "--property_columns", "LogP", "TPSA", "MW", "--scaffold_column", "auto", "--num_epochs", "1",
        "--device", "cpu", "--sample_every", "0"])
    finetune_conditional.main([
        "--pretrained_model", models["mixed"], "--smiles_file", data_csv, "--output_model", out,
        "--compute_properties", "QED", "--property_columns", "LogP", "--binary_property_columns", "active",
        "--bin_property_columns", "binary_activity", "--num_epochs", "1", "--device", "cpu", "--sample_every", "0",
        "--augment_actives_multiplier", "3"])
    assert Model.load(out).bin_edges == Model.load(models["mixed"]).bin_edges


@pytest.mark.parametrize("extra", [
    ["--cell_type", "lstm", "--randomize_smiles", "--num_workers", "2"],
    ["--compute_properties", "MW", "--conditioning_dropout_mode", "example", "--no_learned_null"],
    ["--compute_properties", "MW", "QED", "--rnn_input_size", "96", "--early_stopping_metric", "unconditional"],
    ["--binary_property_columns", "active", "--augment_actives_multiplier", "2", "--conditioning_dropout_mode", "batch"],
    ["--scaffold_column", "auto", "--scaffold_fp_type", "MACCS"],
    ["--blank_binary_conditioning", "2", "--augment_multiplier", "2"],
])
def test_training_variants(tmp_path, extra):
    path = train(tmp_path, "v", *extra)
    m = Model.load(path)
    _, smiles, nlls = m.sample(8)
    assert len(smiles) == 8 and torch.isfinite(nlls).all()


def test_training_errors(tmp_path):
    with pytest.raises(ValueError, match="blank"):
        train(tmp_path, "e", "--blank_conditioning", "1", "--property_columns", "LogP")
    with pytest.raises(ValueError, match="not found"):
        train(tmp_path, "e", "--property_columns", "nope")
    with pytest.raises(ValueError, match="only 0 and 1"):
        train(tmp_path, "e", "--binary_property_columns", "LogP")
    with pytest.raises(SystemExit):
        train(tmp_path, "e", "--no_uncond_val", "--early_stopping_metric", "unconditional")


def test_configs_parse(monkeypatch):
    """Every shipped config parses and only uses keys its script accepts."""
    import tomllib

    import finetune_conditional
    import sample
    import train_conditional
    import validate_conditioning

    entry = {"train": (train_conditional, "train_conditional"), "pretrain": (train_conditional, "train_conditional"),
             "finetune": (finetune_conditional, "finetune_conditional"), "sample": (sample, "sample_molecules"),
             "validate": (validate_conditioning, "validate")}
    for path in sorted((ROOT / "configs").glob("*.toml")):
        module, func = entry[path.stem.split("_")[0]]
        captured = []
        monkeypatch.setattr(module, func, captured.append)
        module.main(["--config", str(path)])
        args = vars(captured[0])
        keys = set()
        for value in tomllib.loads(path.read_text()).values():
            keys |= set(value) if isinstance(value, dict) else set()
        assert keys <= set(args), (path.name, keys - set(args))
