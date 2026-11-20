"""Model wrapper: vocabulary, likelihood, sampling and checkpointing."""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .conditioning import ConditioningConfig
from .rnn import RNN
from .vocabulary import SMILESTokenizer, Vocabulary, START_TOKEN, STOP_TOKEN

logger = logging.getLogger(__name__)


class Model:
    """Conditional chemical language model.

    The fitted property scaler, bin edges and property names are stored in the
    checkpoint so sampling does not need any side files.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        tokenizer: SMILESTokenizer,
        network_params: Optional[dict] = None,
        conditioning_config: Optional[ConditioningConfig] = None,
        max_sequence_length: int = 256,
        mode: str = "training",
        device: torch.device = torch.device("cpu"),
        scaler=None,
        bin_edges: Optional[dict] = None,
        bin_occupancy: Optional[dict] = None,
        property_names: Optional[List[str]] = None,
    ):
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.device = device
        self.conditioning_config = conditioning_config or ConditioningConfig()
        self.scaler = scaler
        self.bin_edges = bin_edges or {}
        self.bin_occupancy = bin_occupancy or {}
        self.property_names = list(property_names or [])

        self.network = RNN(
            len(vocabulary), **(network_params or {}), conditioning_config=self.conditioning_config
        ).to(device)
        self._nll_loss = nn.NLLLoss(reduction="none")
        self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        if mode == "training":
            self.network.train()
        elif mode == "inference":
            self.network.eval()
        else:
            raise ValueError(f"Invalid model mode '{mode}'")

    def parameters(self):
        return self.network.parameters()

    def likelihood(
        self,
        sequences: torch.Tensor,
        properties: Optional[torch.Tensor] = None,
        scaffold_fps: Optional[torch.Tensor] = None,
        property_mask: Optional[torch.Tensor] = None,
        scaffold_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-sequence negative log-likelihood under teacher forcing. Returns (B,)."""
        logits, _ = self.network(
            sequences[:, :-1],
            properties=properties,
            scaffold_fps=scaffold_fps,
            property_mask=property_mask,
            scaffold_mask=scaffold_mask,
        )
        log_probs = logits.log_softmax(dim=2)
        return self._nll_loss(log_probs.transpose(1, 2), sequences[:, 1:]).sum(dim=1)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 128,
        properties: Optional[torch.Tensor] = None,
        scaffold_fps: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        property_mask: Optional[torch.Tensor] = None,
        scaffold_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """Autoregressive sampling. Returns (sequences, smiles, nlls).

        Temperature only changes which token is drawn; the NLL is always taken
        from the untempered distribution.
        """
        start_idx = self.vocabulary[START_TOKEN]
        stop_idx = self.vocabulary[STOP_TOKEN]
        tokens = torch.full((batch_size,), start_idx, dtype=torch.long, device=self.device)
        sequences = [tokens.unsqueeze(1)]
        nlls = torch.zeros(batch_size, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        hidden = None

        for _ in range(self.max_sequence_length - 1):
            logits, hidden = self.network(
                tokens.unsqueeze(1),
                hidden,
                properties=properties,
                scaffold_fps=scaffold_fps,
                property_mask=property_mask,
                scaffold_mask=scaffold_mask,
            )
            logits = logits.squeeze(1)
            probs = (logits / temperature).softmax(dim=1)
            tokens = torch.multinomial(probs, 1).view(-1).masked_fill(finished, stop_idx)
            sequences.append(tokens.unsqueeze(1))
            nll = self._nll_loss(logits.log_softmax(dim=1), tokens)
            nlls += nll.masked_fill(finished, 0.0)
            finished |= tokens == stop_idx
            if finished.all():
                break

        seqs = torch.cat(sequences, dim=1)
        smiles = [
            self.tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in seqs.cpu().numpy()
        ]
        return seqs, smiles, nlls

    def split_property_names(self) -> Tuple[List[str], List[str], List[str]]:
        """Property names split into (scalar, binary, binned); empty if not recorded."""
        cc = self.conditioning_config
        names = self.property_names
        if len(names) != cc.num_properties:
            return [], [], []
        a = cc.num_scalar_properties
        b = a + cc.num_binary_properties
        return names[:a], names[a:b], names[b:]

    def save(self, path: str) -> None:
        torch.save(
            {
                "vocabulary": self.vocabulary.get_dictionary(),
                "max_sequence_length": self.max_sequence_length,
                "network": self.network.state_dict(),
                "network_params": self.network.get_params(),
                "conditioning": self.conditioning_config.as_dict(),
                "scaler": self.scaler,
                "bin_edges": self.bin_edges,
                "bin_occupancy": self.bin_occupancy,
                "property_names": self.property_names,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, mode: str = "inference", device=torch.device("cpu")) -> "Model":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cc = ConditioningConfig.from_dict(ckpt.get("conditioning") or {})
        state = ckpt["network"]

        expected_nulls = []
        if cc.has_property_conditioning:
            expected_nulls.append("_null_property_embedding")
        if cc.has_scaffold_conditioning:
            expected_nulls.append("_null_scaffold_embedding")
        if cc.use_learned_null and any(k not in state for k in expected_nulls):
            logger.warning("Checkpoint has no learned null embedding; using zeros instead.")
            cc.use_learned_null = False

        model = cls(
            vocabulary=Vocabulary.load_from_dictionary(ckpt["vocabulary"]),
            tokenizer=SMILESTokenizer(),
            network_params=ckpt.get("network_params"),
            conditioning_config=cc,
            max_sequence_length=ckpt.get("max_sequence_length", 256),
            mode=mode,
            device=device,
            scaler=ckpt.get("scaler"),
            bin_edges=ckpt.get("bin_edges"),
            bin_occupancy=ckpt.get("bin_occupancy"),
            property_names=ckpt.get("property_names"),
        )
        model.network.load_state_dict(state)
        return model
