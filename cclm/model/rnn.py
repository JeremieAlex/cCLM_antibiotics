"""GRU/LSTM backbone with property and scaffold conditioning.

Based on the REINVENT4 RNN prior. The conditioning embeddings are concatenated
to the token embedding at every step.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditioning import ConditioningConfig, MixedPropertyEmbedder, ScaffoldEmbedder


class RNN(nn.Module):
    def __init__(
        self,
        voc_size: int,
        layer_size: int = 512,
        num_layers: int = 3,
        cell_type: str = "gru",
        embedding_layer_size: int = 256,
        dropout: float = 0.0,
        layer_normalization: bool = False,
        conditioning_config: Optional[ConditioningConfig] = None,
    ):
        super().__init__()
        self._layer_size = layer_size
        self._embedding_layer_size = embedding_layer_size
        self._num_layers = num_layers
        self._cell_type = cell_type.lower()
        self._dropout = dropout
        self._layer_normalization = layer_normalization
        if self._cell_type not in ("gru", "lstm"):
            raise ValueError('cell_type must be "gru" or "lstm"')

        cc = conditioning_config or ConditioningConfig()
        self._cc = cc
        self._embedding = nn.Embedding(voc_size, embedding_layer_size)

        self._property_embedder = None
        self._scaffold_embedder = None
        if cc.has_property_conditioning:
            self._property_embedder = MixedPropertyEmbedder(
                cc.num_scalar_properties,
                cc.num_binary_properties,
                cc.condition_embedding_size,
                cc.property_embedder_layers,
                cc.binary_embedder_layers,
                num_binned=cc.num_binned_properties,
                num_bins=cc.num_bins,
                binned_layers=cc.binned_embedder_layers,
            )
            if cc.use_learned_null:
                self._null_property_embedding = nn.Parameter(torch.zeros(cc.condition_embedding_size))
        if cc.has_scaffold_conditioning:
            self._scaffold_embedder = ScaffoldEmbedder(
                cc.scaffold_fp_size, cc.condition_embedding_size, cc.scaffold_embedder_layers
            )
            if cc.use_learned_null:
                self._null_scaffold_embedding = nn.Parameter(torch.zeros(cc.condition_embedding_size))

        rnn_cls = nn.GRU if self._cell_type == "gru" else nn.LSTM
        self._rnn = rnn_cls(
            input_size=embedding_layer_size + cc.total_conditioning_size,
            hidden_size=layer_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self._linear = nn.Linear(layer_size, voc_size)

    def _null(self, name: str, batch_size: int, ref: torch.Tensor) -> torch.Tensor:
        """Learned null embedding, or zeros if the model was built without one."""
        param = getattr(self, name, None)
        if param is not None:
            return param.to(ref.dtype).unsqueeze(0).expand(batch_size, -1)
        return ref.new_zeros(batch_size, self._cc.condition_embedding_size)

    def _property_embedding(self, properties, mask, batch_size, ref):
        if properties is None:
            return self._null("_null_property_embedding", batch_size, ref)
        emb = self._property_embedder(properties, mask=mask)
        if mask is not None:
            # Rows with every property hidden get the null embedding.
            any_visible = (mask > 0).any(dim=1, keepdim=True).to(emb.dtype)
            null = self._null("_null_property_embedding", batch_size, emb)
            emb = any_visible * emb + (1.0 - any_visible) * null
        return emb

    def _scaffold_embedding(self, fps, mask, batch_size, ref):
        if fps is None:
            return self._null("_null_scaffold_embedding", batch_size, ref)
        emb = self._scaffold_embedder(fps)
        if mask is not None:
            m = mask.reshape(batch_size, 1).to(emb.dtype)
            null = self._null("_null_scaffold_embedding", batch_size, emb)
            emb = m * emb + (1.0 - m) * null
        return emb

    def forward(
        self,
        input_vector: torch.Tensor,
        hidden_state=None,
        properties: Optional[torch.Tensor] = None,
        scaffold_fps: Optional[torch.Tensor] = None,
        property_mask: Optional[torch.Tensor] = None,
        scaffold_mask: Optional[torch.Tensor] = None,
    ):
        """
        :param input_vector: (B, T) token indices
        :param properties: (B, num_properties) or None for the null embedding
        :param scaffold_fps: (B, fp_size) or None for the null embedding
        :param property_mask: (B, num_properties), 1 = visible, 0 = hidden
        :param scaffold_mask: (B,) or (B, 1)
        :returns: (logits, hidden_state)
        """
        batch_size, seq_len = input_vector.size()
        if property_mask is not None and (properties is None or self._property_embedder is None):
            raise ValueError("property_mask needs properties and a property-conditioned model")
        if scaffold_mask is not None and (scaffold_fps is None or self._scaffold_embedder is None):
            raise ValueError("scaffold_mask needs scaffold_fps and a scaffold-conditioned model")

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size)

        embedded = self._embedding(input_vector)
        parts = [embedded]
        if self._property_embedder is not None:
            parts.append(self._property_embedding(properties, property_mask, batch_size, embedded))
        if self._scaffold_embedder is not None:
            parts.append(self._scaffold_embedding(scaffold_fps, scaffold_mask, batch_size, embedded))
        if len(parts) > 1:
            cond = [p.unsqueeze(1).expand(-1, seq_len, -1) for p in parts[1:]]
            embedded = torch.cat([embedded] + cond, dim=-1)

        output, hidden_state = self._rnn(embedded, hidden_state)
        if self._layer_normalization:
            output = F.layer_norm(output, output.size()[1:])
        logits = self._linear(output)
        return logits, hidden_state

    def _init_hidden(self, batch_size: int):
        size = (self._num_layers, batch_size, self._layer_size)
        dev = self._embedding.weight.device
        if self._cell_type == "gru":
            return torch.zeros(*size, device=dev)
        return (torch.zeros(*size, device=dev), torch.zeros(*size, device=dev))

    def get_params(self) -> dict:
        return {
            "dropout": self._dropout,
            "layer_size": self._layer_size,
            "num_layers": self._num_layers,
            "cell_type": self._cell_type,
            "embedding_layer_size": self._embedding_layer_size,
            "layer_normalization": self._layer_normalization,
        }
