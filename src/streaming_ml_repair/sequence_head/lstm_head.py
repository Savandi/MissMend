"""Sequence rescue head for the hybrid framework.

A deliberately minimal 1-layer LSTM that predicts the activity at the current
position from the per-case prefix of length N. Used only on the rescue path
(invoked when the cluster matcher's composite confidence falls below alpha).

Two input modes are supported behind a runtime toggle:

- 'activity_id'  : prefix is a sequence of activity-id tokens; the head
    learns its own activity embeddings. This is the clean control-flow-only
    signal that complements the cluster matcher's multi-perspective view.
- 'z_latent'     : prefix is a sequence of SDAE latent vectors (already
    computed by the pipeline for the cluster matcher). This is the ablation
    arm; in principle helps when IoT context is informative for sequence
    structure, in practice expected to inherit the IoT-sparsity problem on
    Cotton/Vienna where the rescue is most needed.

Special tokens for the activity-id input mode:
    0 = <PAD>   (left-padding for short prefixes)
    1 = <MISSING>   (used when a prior event in the prefix is UNRECOVERED_ML)
    2..  activity ids

The output is a softmax over the activity vocabulary, with PAD/MISSING masked
out of the prediction so the head never argmaxes to a non-activity token.
"""
from __future__ import annotations
import torch
import torch.nn as nn

PAD_TOKEN_ID = 0
MISSING_TOKEN_ID = 1

class SequenceRescueHead(nn.Module):

    def __init__(
        self,
        activity_vocab_size: int,
        latent_dim: int | None = None,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        input_mode: str = "activity_id",
        num_layers: int = 1,
    ):
        """
        Args:
            activity_vocab_size: number of activity ids (including <PAD>=0
                and <MISSING>=1). The output softmax is over this vocab.
            latent_dim: dimensionality of the SDAE latent vectors. Required
                only when input_mode == 'z_latent'.
            embedding_dim: width of the activity-id embedding. Ignored in
                z_latent mode.
            hidden_dim: LSTM hidden width.
            dropout: applied to the LSTM input and to the pre-output features.
            input_mode: 'activity_id' (default) or 'z_latent'.
        """
        super().__init__()
        if input_mode not in ("activity_id", "z_latent"):
            raise ValueError(f"input_mode must be 'activity_id' or 'z_latent', got {input_mode!r}")
        if input_mode == "z_latent" and latent_dim is None:
            raise ValueError("latent_dim must be provided when input_mode='z_latent'")

        self.input_mode = input_mode
        self.activity_vocab_size = activity_vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

        if input_mode == "activity_id":
            self.embed = nn.Embedding(
                num_embeddings=activity_vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=PAD_TOKEN_ID,
            )
            input_size = embedding_dim
        else:
            self.embed = None
            input_size = latent_dim

        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, activity_vocab_size)

    def forward(self, x):
        """
        Args:
            x: in activity_id mode, (B, N) long tensor of activity ids.
               in z_latent mode, (B, N, latent_dim) float tensor.
        Returns:
            logits: (B, activity_vocab_size) logits at the *last* LSTM output
                (the position immediately following the prefix; this is what
                the rescue head predicts for).
        """
        if self.input_mode == "activity_id":
            x_in = self.embed(x)
        else:
            x_in = x

        x_in = self.dropout(x_in)
        out, _ = self.lstm(x_in)
        last = out[:, -1, :]
        last = self.dropout(last)
        logits = self.classifier(last)
        return logits

    def predict(self, x):
        """Return (pred_id, pred_conf) with PAD and MISSING masked out so the
        head never argmaxes to a non-activity token. pred_conf is the softmax
        probability mass on the argmax position (uncalibrated; calibrated
        separately by the temperature scaler before being compared to
        alpha_seq).
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            probs_clipped = probs.clone()
            probs_clipped[:, PAD_TOKEN_ID] = 0.0
            probs_clipped[:, MISSING_TOKEN_ID] = 0.0
            pred_id = torch.argmax(probs_clipped, dim=-1)
            pred_conf = probs_clipped.gather(1, pred_id.unsqueeze(1)).squeeze(1)
        return pred_id, pred_conf, probs
