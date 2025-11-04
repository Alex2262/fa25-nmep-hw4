from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self, num_heads: int, embedding_dim: int, qk_length: int, value_length: int
    ):
        """
        The Multi-Head Attention layer will take in Q, K, and V
        matrices and will output an attention matrix of shape <TODO>.

        First, Q, K, and V should be projected to have
        a shape of (B, T, C) where C = num_heads * qk_length
        (OR value_length). You are then expected to split
        the C dimension into num_heads different heads, each
        with shape (B, T, vec_length).

        Next, you will compute the scaled dot-product attention
        between Q, K, and V.

        Finally, you will concatenate the heads and project the
        output to have a shape of (B, T, C).

        Check out the `masked_fill` method in PyTorch to help
        you implement the masking step!
        """
        super().__init__()

        self.num_heads = num_heads
        self.qk_length = qk_length
        self.value_length = value_length

        qk_dim = num_heads * qk_length
        v_dim = num_heads * value_length

        self.weights_q = nn.Linear(embedding_dim, qk_dim)
        self.weights_k = nn.Linear(embedding_dim, qk_dim)
        self.weights_v = nn.Linear(embedding_dim, v_dim)
        self.weights_o = nn.Linear(v_dim, embedding_dim)

    def split_heads(self, x: torch.Tensor, vec_length: int) -> torch.Tensor:
        """
        Split the C dimension of the input tensor into num_heads
        different heads, each with shape (B, T, vec_length).
        Hint: check out the `view` and 'permute` methods in PyTorch to help
        you reshape the tensor.

        Args:
            x: torch.Tensor of shape (B, T, C), where C = num_heads * vec_length
            vec_length: int, the length of the query/key/value vectors

        Returns:
            torch.Tensor of shape (B, num_heads, T, vec_length)
        """
        B, T, C = x.size()

        # print("SPLIT HEADS:", x.size(), self.num_heads, vec_length)

        assert C // self.num_heads == vec_length, (
            "Input tensor does not have the correct shape for splitting."
        )

        reshaped = x.view(B, T, self.num_heads, vec_length)
        reshaped = torch.permute(reshaped, (0, 2, 1, 3))

        assert(reshaped.size() == (B, self.num_heads, T, vec_length))

        return reshaped

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine the num_heads different heads into a single tensor.
        Hint: check out the `contiguous` method in PyTorch to help
        you reshape the tensor.

        Args:
            x: torch.Tensor of shape (B, num_heads, T, vec_length)

        Returns:
            torch.Tensor of shape (B, T, num_heads * vec_length)
        """

        # print("COMBINE HEADS INP:", x.shape)

        B, num_heads, T, vec_length = x.size()

        x = torch.permute(x, (0, 2, 1, 3)).contiguous()
        y = x.view(B, T, num_heads * vec_length)

        assert(y.size() == (B, T, num_heads * vec_length))

        return y

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the scaled dot-product attention given Q, K, and V.
        This is where the pad_mask and causal_mask are applied.

        Args:
            Q: torch.Tensor of shape (B, num_heads, T, qk_length)
            K: torch.Tensor of shape (B, num_heads, T, qk_length)
            V: torch.Tensor of shape (B, num_heads, T, value_length)
            mask: Optional boolean torch.Tensor, broadcastable to (B, num_heads, T, T).
        """

        B, num_heads, T, qk_length = Q.size()

        inner = Q @ torch.transpose(K, -2, -1) / (self.qk_length ** 0.5)
        if mask is not None:
            inner = inner.masked_fill(mask == 1, float('-inf'))

        # print("INNER SHAPE", inner.shape)

        out = torch.softmax(inner, dim=3) @ V

        # print("OUT SHAPE", out.shape)

        assert(out.size() == (B, num_heads, T, self.value_length))

        return out

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        The forward pass of the Multi-Head Attention layer.

        Args:
            Q: torch.Tensor of shape (B, T, C)
            K: torch.Tensor of shape (B, T, C)
            V: torch.Tensor of shape (B, T, C)
            mask: Optional torch.Tensor of shape (B, T, T) or None

        Returns:
            torch.Tensor of shape (B, T, C)
        """

        B, T, C = Q.size()

        Qs = self.weights_q(Q)
        Ks = self.weights_k(K)
        Vs = self.weights_v(V)

        Qh = self.split_heads(Qs, self.qk_length)
        Kh = self.split_heads(Ks, self.qk_length)
        Vh = self.split_heads(Vs, self.value_length)

        out = self.scaled_dot_product_attention(Qh, Kh, Vh, mask)

        out = self.combine_heads(out)

        out = self.weights_o(out)

        assert(out.size() == (B, T, C))

        return out



class FeedForwardNN(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        """
        The Feed-Forward Neural Network layer will take in
        an input tensor of shape (B, T, C) and will output
        a tensor of the same shape.

        The FFNN will have two linear layers, with a ReLU
        activation function in between.

        Args:
            hidden_dim: int, the size of the hidden layer
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Define any layers you'll need in the forward pass
        self.l1 = nn.Linear(embedding_dim, hidden_dim)
        self.act = nn.GELU()
        self.l2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the FeedForwardNN.
        """

        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)

        return x
