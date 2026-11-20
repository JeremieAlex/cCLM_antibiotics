import logging
import random
from typing import Callable, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

logger = logging.getLogger(__name__)

# C, N, O, F, S, Cl, Br, I
ALLOWED_ELEMENTS = {6, 7, 8, 9, 16, 17, 35, 53}


def parse_generated_smiles(smiles: str) -> Optional[Chem.Mol]:
    """RDKit mol for a sampled SMILES, or None if it is empty or invalid."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return mol if mol is not None and mol.GetNumAtoms() > 0 else None


def filter_mol(mol, max_heavy_atoms: int = 100, min_heavy_atoms: int = 10) -> bool:
    if mol is None:
        return False
    if not min_heavy_atoms < mol.GetNumHeavyAtoms() < max_heavy_atoms:
        return False
    return all(a.GetAtomicNum() in ALLOWED_ELEMENTS for a in mol.GetAtoms())


def canonicalize_and_filter(
    smiles_list: List[str], max_heavy_atoms: int = 100, min_heavy_atoms: int = 10
) -> Tuple[List[str], List[int]]:
    """Canonical (isomeric) SMILES of the molecules that pass ``filter_mol``, and their indices."""
    canonical, kept = [], []
    for i, smi in enumerate(tqdm(smiles_list, desc="Canonicalizing", unit="mol")):
        mol = Chem.MolFromSmiles(smi)
        if filter_mol(mol, max_heavy_atoms, min_heavy_atoms):
            canonical.append(Chem.MolToSmiles(mol))
            kept.append(i)
    return canonical, kept


def get_scaffold(smiles: str) -> Optional[str]:
    """Murcko scaffold SMILES ("" for acyclic molecules), or None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return None


def random_smiles(mol) -> str:
    return Chem.MolToRandomSmilesVect(mol, 1, randomSeed=random.getrandbits(31))[0]


def augment_smiles(
    smiles_list: List[str],
    multiplier: int,
    properties: Optional[np.ndarray] = None,
    scaffold_fps: Optional[np.ndarray] = None,
    is_valid: Optional[Callable[[str], bool]] = None,
    max_retries: int = 5,
):
    """``multiplier`` SMILES per molecule: the canonical one followed by random ones.

    A random SMILES rejected by ``is_valid`` is redrawn up to ``max_retries``
    times before falling back to the canonical form. Property and fingerprint
    rows are repeated to match. Returns (smiles, properties, scaffold_fps).
    """
    out, index = [], []
    for i, smi in enumerate(tqdm(smiles_list, desc="Augmenting SMILES", unit="mol")):
        out.append(smi)
        index.append(i)
        mol = Chem.MolFromSmiles(smi)
        for _ in range(multiplier - 1):
            variant = smi
            for _ in range(max_retries if mol is not None else 0):
                cand = random_smiles(mol)
                if is_valid is None or is_valid(cand):
                    variant = cand
                    break
            out.append(variant)
            index.append(i)
    return (
        out,
        properties[index] if properties is not None else None,
        scaffold_fps[index] if scaffold_fps is not None else None,
    )
