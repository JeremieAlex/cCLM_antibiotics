from typing import List, Optional

import numpy as np
import torch
import torch.utils.data as tud
from rdkit import Chem

from cclm.model.vocabulary import SMILESTokenizer, Vocabulary
from cclm.utils.chemistry import random_smiles
from cclm.utils.device import pad_sequences


def vocab_checker(vocabulary: Vocabulary, tokenizer: SMILESTokenizer):
    """Function that tells whether a SMILES can be encoded with ``vocabulary``."""
    def ok(smi: str) -> bool:
        return all(t in vocabulary for t in tokenizer.tokenize(smi))
    return ok


def split_train_val(smiles, properties, scaffold_fps, val_split: float):
    """Random split by molecule. Returns ((smiles, props, fps), (smiles, props, fps))."""
    n_train = int((1.0 - val_split) * len(smiles))
    perm = torch.randperm(len(smiles)).numpy()

    def take(idx):
        idx = np.sort(idx)
        return (
            [smiles[i] for i in idx],
            properties[idx] if properties is not None else None,
            scaffold_fps[idx] if scaffold_fps is not None else None,
        )

    return take(perm[:n_train]), take(perm[n_train:])


class ConditionalDataset(tud.Dataset):
    """Yields (tokens, properties or None, scaffold fingerprint or None).

    With ``randomize=True`` each item is a fresh random SMILES of the molecule,
    falling back to the stored SMILES if the random one has unknown tokens.
    """

    def __init__(
        self,
        smiles: List[str],
        vocabulary: Vocabulary,
        tokenizer: SMILESTokenizer,
        properties: Optional[np.ndarray] = None,
        scaffold_fps: Optional[np.ndarray] = None,
        randomize: bool = False,
    ):
        self.smiles = smiles
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.properties = properties
        self.scaffold_fps = scaffold_fps
        self.randomize = randomize
        self._in_vocab = vocab_checker(vocabulary, tokenizer)

    def __len__(self):
        return len(self.smiles)

    def _smiles(self, i: int) -> str:
        smi = self.smiles[i]
        if self.randomize:
            mol = Chem.MolFromSmiles(smi)
            for _ in range(5 if mol is not None else 0):
                cand = random_smiles(mol)
                if self._in_vocab(cand):
                    return cand
        return smi

    def __getitem__(self, i: int):
        tokens = self.vocabulary.encode(self.tokenizer.tokenize(self._smiles(i)))
        props = None if self.properties is None else torch.from_numpy(self.properties[i])
        fps = None if self.scaffold_fps is None else torch.from_numpy(self.scaffold_fps[i])
        return torch.from_numpy(tokens), props, fps

    @staticmethod
    def collate(batch):
        seqs, props, fps = zip(*batch)
        return (
            pad_sequences(list(seqs)),
            torch.stack(props) if props[0] is not None else None,
            torch.stack(fps) if fps[0] is not None else None,
        )
