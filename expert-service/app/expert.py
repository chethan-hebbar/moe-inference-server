import torch
import torch.nn as nn

class Expert(nn.Module):
    """
    A simple feed-forward network, which will serve as an 'expert' in our MoE layer.
    """
    def __init__(self, d_model, d_hidden):
        """
        Args:
            d_model (int): The input and output dimension of the model.
            d_hidden (int): The dimension of the hidden layer.
        """
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model)
        )

    def forward(self, x):
        """
        Forward pass for the expert.

        Args:
            x (torch.Tensor): The input tensor. Shape: [..., d_model]

        Returns:
            torch.Tensor: The output tensor. Shape: [..., d_model]
        """
        return self.net(x)
