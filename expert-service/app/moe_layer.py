import torch
import torch.nn as nn
import torch.nn.functional as F

from expert import Expert
from gating import Gating

class MoELayer(nn.Module):
    """
    A Mixture of Experts layer.
    """
    def __init__(self, d_model, d_hidden, num_experts, top_k):
        """
        Args:
            d_model (int): The dimension of the input and output.
            d_hidden (int): The hidden dimension of each expert FFN.
            num_experts (int): The total number of experts.
            top_k (int): The number of experts to route each token to.
        """
        super(MoELayer, self).__init__()

        # Basic validation
        if top_k > num_experts:
            raise ValueError("top_k must be less than or equal to num_experts")

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # Instantiate the experts and the gating network
        self.experts = nn.ModuleList([Expert(d_model, d_hidden) for _ in range(num_experts)])
        self.gating = Gating(d_model, num_experts)

    def compute_load_balancing_loss(self, gating_logits):
        """
        Computes the load balancing loss for the MoE layer.
        This loss encourages the gating network to distribute tokens evenly across experts.

        Args:
            gating_logits (torch.Tensor): The raw logits from the gating network.
                                      Shape: [batch_size * seq_len, num_experts]

        Returns:
            torch.Tensor: A single scalar value representing the load balancing loss.
        """
        #
        # The formula is: alpha * sum(f_i * P_i) for i in experts
        # f_i = fraction of tokens sent to expert i
        # P_i = average probability (gating value) for expert i over tokens sent to it
        #

        # Calculate P_i: softmax over all logits
        gating_probs = F.softmax(gating_logits, dim=-1)

        # Calculate f_i: mean of the one-hot encoding of the chosen expert
        # For top-k > 1, this is more complex. A simplification is to look at the prob distribution.
        # We can calculate the fraction of the "load" each expert gets.
        f_i = gating_probs.mean(dim=0) # Shape: [num_experts]

        # Calculate P_i: The mean of the probabilities assigned to each expert across all tokens.
        P_i = gating_probs.mean(dim=0) # In this simplified case, f_i and P_i are the same.
                                   # More advanced implementations differ.

        # The loss is the dot product of these two vectors, scaled by the number of experts.
        # This encourages the product (and thus both f_i and P_i) to be uniform.
        loss = self.num_experts * torch.sum(f_i * P_i)
        return loss

    def forward(self, x):
        """
        Forward pass for the MoE layer.

        Args:
            x (torch.Tensor): Input tensor. Shape: [batch_size, seq_len, d_model]

        Returns:
            (This will be implemented tomorrow)
        """
        # Reshape input for gating: [batch_size * seq_len, d_model]
        # This treats each token independently.
        batch_size, seq_len, d_model = x.shape
        x_reshaped = x.view(-1, d_model)

        # Get gating logits: [batch_size * seq_len, num_experts]
        gating_logits = self.gating(x_reshaped)

        # Get the top-k experts and their scores (gating values)
        # The scores are softmax-normalized logits for the top-k experts.
        # top_k_gating_values shape: [batch_size * seq_len, top_k]
        # top_k_indices shape: [batch_size * seq_len, top_k]
        top_k_gating_values, top_k_indices = torch.topk(gating_logits, self.top_k, dim=-1)

        # Apply softmax to the top-k logits to get weights
        top_k_gating_values = F.softmax(top_k_gating_values, dim=-1)

        # Create a flat tensor of token indices
        # This will be [0, 0, 1, 1, 2, 2, ...] for top_k=2
        # It helps us track which output belongs to which original token.
        token_indices = torch.arange(x_reshaped.size(0)).repeat_interleave(self.top_k)

        # Create a flat tensor of the chosen expert indices for all tokens
        flat_expert_indices = top_k_indices.flatten()

        # Create our dispatch mask. It's a binary matrix of shape
        # [batch_size * seq_len, num_experts].
        # Entry (i, j) is 1 if token i is routed to expert j, and 0 otherwise.
        dispatch_mask = torch.zeros(x_reshaped.size(0), self.num_experts, device=x.device).bool()
        dispatch_mask.scatter_(1, top_k_indices, True)

        # The final output tensor, initialized to zeros
        final_output = torch.zeros_like(x_reshaped)

        # Now, iterate through each expert.
        for i in range(self.num_experts):
          # Find the tokens that are routed to this expert
          expert_mask = dispatch_mask[:, i]

          # If no tokens are routed to this expert, skip it.
          if not expert_mask.any():
            continue

          # Get the indices of the tokens for this expert
          token_ids_for_expert = expert_mask.nonzero(as_tuple=True)[0]

          # Get the actual input tokens for this expert
          inputs_for_expert = x_reshaped[token_ids_for_expert]

          # Pass the tokens through the expert
          expert_output = self.experts[i](inputs_for_expert)

          # Find the gating values associated with these tokens for this expert
          gating_values_for_expert = top_k_gating_values[dispatch_mask[:, i]]

          # The gating values tensor is currently [num_tokens_for_expert, top_k].
          # We need to find which of the top_k is our current expert 'i'.
          # We create a mask for this.
          k_mask = (top_k_indices[expert_mask] == i)

          # Apply the mask to get the single correct gating value for each token.
          correct_gating_values = gating_values_for_expert[k_mask]

          # Multiply the expert output by the gating values (element-wise)
          weighted_output = expert_output * correct_gating_values.unsqueeze(-1)

          # Add the weighted output to the final output tensor at the correct positions.
          # This is the "combine" step. We use index_add_ for an efficient scatter-add.
          final_output.index_add_(0, token_ids_for_expert, weighted_output)

        # Reshape the final output back to the original input shape
        return final_output.view(batch_size, seq_len, d_model), gating_logits
