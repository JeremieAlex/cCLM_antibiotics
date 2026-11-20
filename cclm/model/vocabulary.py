"""Vocabulary and regex SMILES tokenizer (from REINVENT4).

Bracket atoms, two-digit ring closures (%NN), Br and Cl are single tokens.
"""

import re
from typing import Dict, List

import numpy as np

START_TOKEN = "^"
STOP_TOKEN = "$"

_REGEXPS = [
    re.compile(r"(\[[^\]]*\])"),
    re.compile(r"(%\d{2})"),
    re.compile(r"(Br|Cl)"),
]


class Vocabulary:
    """Two-way token <-> index mapping."""

    def __init__(self):
        self._tokens: Dict = {}
        self._next_id = 0

    def add(self, token: str) -> int:
        if token not in self._tokens:
            self._tokens[token] = self._next_id
            self._tokens[self._next_id] = token
            self._next_id += 1
        return self._tokens[token]

    def encode(self, tokens: List[str]) -> np.ndarray:
        try:
            return np.array([self._tokens[t] for t in tokens], dtype=np.int64)
        except KeyError as e:
            raise KeyError(f"Token {e} is not in the vocabulary") from None

    def decode(self, indices) -> List[str]:
        return [self._tokens[int(i)] for i in indices]

    def tokens(self) -> List[str]:
        return [t for t in self._tokens if isinstance(t, str)]

    def get_dictionary(self) -> dict:
        return {"tokens": {t: self._tokens[t] for t in self.tokens()}}

    @classmethod
    def load_from_dictionary(cls, d: dict) -> "Vocabulary":
        vocab = cls()
        for token, idx in sorted(d["tokens"].items(), key=lambda kv: kv[1]):
            vocab._tokens[str(token)] = int(idx)
            vocab._tokens[int(idx)] = str(token)
            vocab._next_id = max(vocab._next_id, int(idx) + 1)
        return vocab

    def __getitem__(self, token_or_id):
        return self._tokens[token_or_id]

    def __contains__(self, token_or_id):
        return token_or_id in self._tokens

    def __len__(self):
        return len(self._tokens) // 2

    def __eq__(self, other):
        return isinstance(other, Vocabulary) and self._tokens == other._tokens

    def __repr__(self):
        return f"Vocabulary({len(self)} tokens)"


class SMILESTokenizer:
    def tokenize(self, smiles: str, with_begin_and_end: bool = True) -> List[str]:
        tokens = _split(smiles, 0)
        if with_begin_and_end:
            tokens = [START_TOKEN] + tokens + [STOP_TOKEN]
        return tokens

    def untokenize(self, tokens: List[str]) -> str:
        out = []
        for token in tokens:
            if token == STOP_TOKEN:
                break
            if token != START_TOKEN:
                out.append(token)
        return "".join(out)


def _split(text: str, level: int) -> List[str]:
    if level == len(_REGEXPS):
        return list(text)
    tokens = []
    for i, piece in enumerate(_REGEXPS[level].split(text)):
        if i % 2 == 0:
            tokens += _split(piece, level + 1)
        else:
            tokens.append(piece)
    return tokens


def create_vocabulary(smiles_list: List[str], tokenizer: SMILESTokenizer) -> Vocabulary:
    tokens = set()
    for smi in smiles_list:
        tokens.update(tokenizer.tokenize(smi, with_begin_and_end=False))
    vocab = Vocabulary()
    for t in [STOP_TOKEN, START_TOKEN] + sorted(tokens):
        vocab.add(t)
    return vocab
