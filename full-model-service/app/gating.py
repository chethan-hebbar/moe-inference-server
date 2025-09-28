import torch
import torch.nn as nn

class Gating(nn.Module):
    """
    A simple linear layer that acts as the gating mechanism in the MoE.
    It decides which experts to route the tokens to.
    """
    def __init__(self, d_model, num_experts):
        """
        Args:
            d_model (int): The dimension of the input tokens.
            num_experts (int): The total number of experts in the MoE layer.
        """
        super(Gating, self).__init__()
        # A single linear layer to produce logits for each expert
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        """
        Forward pass for the gating network.

        Args:
            x (torch.Tensor): The input tensor from the Transformer.
                              Shape: [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: The logits for each expert.
                          Shape: [batch_size, seq_len, num_experts]
        """
        # The output of this layer will be the raw scores (logits) for each expert
        return self.gate(x)