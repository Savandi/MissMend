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
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            probs_clipped = probs.clone()
            probs_clipped[:, PAD_TOKEN_ID] = 0.0
            probs_clipped[:, MISSING_TOKEN_ID] = 0.0
            pred_id = torch.argmax(probs_clipped, dim=-1)
            pred_conf = probs_clipped.gather(1, pred_id.unsqueeze(1)).squeeze(1)
        return pred_id, pred_conf, probs
