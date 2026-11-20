import logging
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem

from cclm.utils.chemistry import get_scaffold

logger = logging.getLogger(__name__)

DEFAULT_SCAFFOLD_FP_TYPE = "Morgan"

# Properties a molecule cannot have below its scaffold's own value.
SCAFFOLD_LOWER_BOUNDED = ("MW", "HeavyAtoms", "Rings", "AromaticRings")


def _canonical(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def murcko_or_self(smiles: str) -> Optional[str]:
    """Canonical Murcko scaffold, or the molecule itself if it has no rings."""
    canonical = _canonical(smiles)
    if canonical is None:
        return None
    scaffold = get_scaffold(canonical)
    return _canonical(scaffold) if scaffold else canonical


def resolve_scaffold(scaffold_smiles: Optional[str] = None, full_molecule: Optional[str] = None):
    """Canonical scaffold from --scaffold_smiles or --full_molecule, or None if neither is given."""
    if scaffold_smiles and full_molecule:
        raise ValueError("Give --scaffold_smiles or --full_molecule, not both.")
    if full_molecule:
        scaffold = murcko_or_self(full_molecule)
        if scaffold is None:
            raise ValueError(f"Invalid --full_molecule SMILES: {full_molecule}")
        logger.info(f"Scaffold of --full_molecule: {scaffold}")
        return scaffold
    if scaffold_smiles:
        given = _canonical(scaffold_smiles)
        if given is None:
            raise ValueError(f"Invalid --scaffold_smiles: {scaffold_smiles}")
        if murcko_or_self(given) != given:
            logger.warning(
                f"{given} is not a Murcko scaffold; the model was trained on Murcko scaffolds only. "
                "Consider passing the molecule with --full_molecule instead."
            )
        return given
    return None


def resolve_scaffold_fp(cc, fp_type: Optional[str] = None, fp_size: Optional[int] = None) -> Tuple[str, int]:
    """Fingerprint type and size recorded in the checkpoint; explicit values must agree."""
    from cclm.training.property_calculator import FingerprintCalculator

    if not cc.has_scaffold_conditioning:
        raise ValueError("This model was trained without scaffold conditioning.")
    trained_type = cc.scaffold_fp_type or DEFAULT_SCAFFOLD_FP_TYPE
    if fp_type and fp_type != trained_type:
        raise ValueError(f"--scaffold_fp_type {fp_type} does not match the checkpoint ({trained_type}).")
    if fp_size is not None and fp_size != cc.scaffold_fp_size:
        raise ValueError(f"--scaffold_fp_size {fp_size} does not match the checkpoint ({cc.scaffold_fp_size}).")
    if FingerprintCalculator(trained_type, cc.scaffold_fp_size).fp_size != cc.scaffold_fp_size:
        raise ValueError(f"{trained_type} fingerprints cannot have {cc.scaffold_fp_size} bits.")
    return trained_type, cc.scaffold_fp_size


def scaffold_fingerprint(scaffold_smiles: str, fp_type: str, fp_size: int) -> np.ndarray:
    from cclm.training.property_calculator import FingerprintCalculator

    if Chem.MolFromSmiles(scaffold_smiles) is None:
        raise ValueError(f"Invalid scaffold SMILES: {scaffold_smiles}")
    return FingerprintCalculator(fp_type, fp_size).compute([scaffold_smiles], progress=False)[0]


def scaffold_retention(smiles_list: List[str], scaffold_smiles: str) -> Tuple[np.ndarray, np.ndarray]:
    """Per molecule: (same Murcko scaffold as the target, target present as a substructure)."""
    target = _canonical(scaffold_smiles)
    if target is None:
        raise ValueError(f"Invalid scaffold SMILES: {scaffold_smiles}")
    query = Chem.MolFromSmiles(target)
    exact = np.zeros(len(smiles_list), dtype=bool)
    substructure = np.zeros(len(smiles_list), dtype=bool)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        exact[i] = murcko_or_self(smi) == target
        substructure[i] = mol.HasSubstructMatch(query)
    return exact, substructure
