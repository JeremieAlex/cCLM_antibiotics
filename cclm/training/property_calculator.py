"""RDKit descriptors and fingerprints selected by name, e.g. "LogP" or "Morgan"."""

import logging
from typing import Callable, Dict, List, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.QED import qed
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _partial_charges(mol):
    AllChem.ComputeGasteigerCharges(mol)
    charges = [a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms()]
    return [c for c in charges if np.isfinite(c)]


PROPERTIES: Dict[str, tuple] = {
    "MW": ("Molecular weight", Descriptors.MolWt),
    "LogP": ("Crippen LogP", Crippen.MolLogP),
    "TPSA": ("Topological polar surface area", Descriptors.TPSA),
    "QED": ("Quantitative estimate of drug-likeness", qed),
    "HBA": ("Hydrogen-bond acceptors", Descriptors.NumHAcceptors),
    "HBD": ("Hydrogen-bond donors", Descriptors.NumHDonors),
    "RotBonds": ("Rotatable bonds", Descriptors.NumRotatableBonds),
    "AromaticRings": ("Aromatic rings", Descriptors.NumAromaticRings),
    "Rings": ("Rings", Descriptors.RingCount),
    "FractionCSP3": ("Fraction of sp3 carbons", Descriptors.FractionCSP3),
    "HeavyAtoms": ("Heavy atoms", Descriptors.HeavyAtomCount),
    "MolRefractivity": ("Crippen molar refractivity", Crippen.MolMR),
    "Lipinski_HBA": ("Lipinski H-bond acceptors", rdMolDescriptors.CalcNumLipinskiHBA),
    "Lipinski_HBD": ("Lipinski H-bond donors", rdMolDescriptors.CalcNumLipinskiHBD),
    "BertzCT": ("Bertz complexity index", Descriptors.BertzCT),
    "LabuteASA": ("Labute approximate surface area", Descriptors.LabuteASA),
    "MaxPartialCharge": ("Max Gasteiger partial charge", lambda m: max(_partial_charges(m))),
    "MinPartialCharge": ("Min Gasteiger partial charge", lambda m: min(_partial_charges(m))),
}


def _bitvect_to_numpy(fp, n_bits):
    arr = np.zeros(n_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _morgan(radius):
    def fp(mol, n_bits):
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        return gen.GetFingerprintAsNumPy(mol).astype(np.float32)
    return fp


FINGERPRINTS: Dict[str, tuple] = {
    "Morgan": ("Morgan, radius 2 (ECFP4)", _morgan(2), 2048),
    "Morgan6": ("Morgan, radius 3 (ECFP6)", _morgan(3), 2048),
    "RDKit": ("RDKit topological fingerprint",
              lambda m, n: _bitvect_to_numpy(Chem.RDKFingerprint(m, fpSize=n), n), 2048),
    "MACCS": ("MACCS keys (always 167 bits)",
              lambda m, n: _bitvect_to_numpy(MACCSkeys.GenMACCSKeys(m), 167), 167),
}


def _safe(func: Callable, mol) -> float:
    if mol is None:
        return np.nan
    try:
        return float(func(mol))
    except Exception:
        return np.nan


class PropertyCalculator:
    def __init__(self, names: List[str]):
        unknown = [n for n in names if n not in PROPERTIES]
        if unknown:
            raise ValueError(f"Unknown properties {unknown}. Available: {', '.join(PROPERTIES)}")
        self.names = list(names)

    def compute(self, smiles_list: List[str], progress: bool = True) -> np.ndarray:
        """(N, num_properties) float32 array; NaN where a value could not be computed."""
        funcs = [PROPERTIES[n][1] for n in self.names]
        out = np.full((len(smiles_list), len(funcs)), np.nan, dtype=np.float32)
        for i, smi in enumerate(tqdm(smiles_list, desc="Computing properties", disable=not progress)):
            mol = Chem.MolFromSmiles(smi)
            out[i] = [_safe(f, mol) for f in funcs]
        return out


class FingerprintCalculator:
    def __init__(self, fp_type: str = "Morgan", n_bits: Optional[int] = None):
        if fp_type not in FINGERPRINTS:
            raise ValueError(f"Unknown fingerprint '{fp_type}'. Available: {', '.join(FINGERPRINTS)}")
        self.name = fp_type
        _, self._func, default_bits = FINGERPRINTS[fp_type]
        self.fp_size = 167 if fp_type == "MACCS" else (n_bits or default_bits)

    def compute(self, smiles_list: List[str], progress: bool = True) -> np.ndarray:
        """(N, fp_size) float32 array; all zeros for unparseable SMILES."""
        out = np.zeros((len(smiles_list), self.fp_size), dtype=np.float32)
        for i, smi in enumerate(tqdm(smiles_list, desc=f"{self.name} fingerprints", disable=not progress)):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                out[i] = self._func(mol, self.fp_size)
        return out


def print_available():
    print("Properties:")
    for name, (desc, _) in PROPERTIES.items():
        print(f"  {name:18s} {desc}")
    print("Fingerprints:")
    for name, (desc, _, _) in FINGERPRINTS.items():
        print(f"  {name:18s} {desc}")
