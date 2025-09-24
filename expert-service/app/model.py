import torch
import torch.nn as nn
import math

# Import the MoELayer you built in Week 2
from moe_layer import MoELayer

# --- Positional Encoding: A standard, non-learnable component ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# --- A Custom Transformer Encoder Layer using our MoELayer ---
class TransformerEncoderLayerWithMoE(nn.Module):
    def __init__(self, d_model, nhead, d_hidden, num_experts, top_k, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.moe_layer = MoELayer(d_model, d_hidden, num_experts, top_k)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # Attention block
        attn_output, _ = self.self_attn(src, src, src)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        # MoE block
        moe_output, gating_logits = self.moe_layer(src)
        src = src + self.dropout(moe_output)
        src = self.norm2(src)
        return src, gating_logits

# --- The Full Classification Model ---
class MoETransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, d_hidden, num_experts, top_k, num_classes, num_layers):
        super(MoETransformerClassifier, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Stack of our custom MoE-enabled encoder layers
        self.transformer_encoder = nn.ModuleList(
            [TransformerEncoderLayerWithMoE(d_model, nhead, d_hidden, num_experts, top_k) for _ in range(num_layers)]
        )

        # The final classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src):
        # src shape: [batch_size, seq_len]
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        all_gating_logits = []

        # Pass through the stack of encoder layers
        for layer in self.transformer_encoder:
            src, gating_logits = layer(src)
            all_gating_logits.append(gating_logits)

        # Pooling: Average the outputs of all tokens in the sequence
        pooled_output = src.mean(dim=1)

        # Final classification
        output_logits = self.classifier(pooled_output)
        return output_logits, all_gating_logits
