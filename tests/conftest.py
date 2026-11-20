import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DATA = ROOT / "tests" / "data" / "small.csv"
TINY = ["--layer_size", "64", "--embedding_size", "32", "--condition_embedding_size", "32",
        "--num_epochs", "1", "--batch_size", "64", "--sample_size", "16", "--device", "cpu",
        "--seed", "0"]


@pytest.fixture(scope="session")
def data_csv():
    return str(DATA)


def train(out_dir, name, *extra):
    import train_conditional
    out = str(Path(out_dir) / f"{name}.ckpt")
    train_conditional.main(["--smiles_file", str(DATA), "--output_model", out, *TINY, *extra])
    return out


@pytest.fixture(scope="session")
def models(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    return {
        "physchem": train(d, "physchem", "--property_columns", "LogP", "TPSA", "MW",
                          "--property_ranges", "2", "4", "50", "110", "300", "450"),
        "mixed": train(d, "mixed", "--compute_properties", "QED", "--property_columns", "LogP",
                       "--binary_property_columns", "active", "--bin_property_columns", "binary_activity",
                       "--num_bins", "5", "--property_ranges", "0.5", "0.9", "1", "4", "1", "1",
                       "--sample_property_bins", "top"),
        "scaffold": train(d, "scaffold", "--property_columns", "LogP", "TPSA", "MW",
                          "--scaffold_column", "auto", "--scaffold_fp_size", "512"),
        "blank": train(d, "blank", "--blank_conditioning", "1"),
    }
