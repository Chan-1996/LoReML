import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRaLinear(nn.Linear):
    """
    Fully Connected layer with on the fly adaptive mask.
    """

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            weight_rank=4,
            alpha=8
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)

        self.weight_rank = weight_rank
        self.alpha = alpha

        # create trainable params
        self.weight_U = nn.Parameter(torch.zeros(out_features, self.weight_rank))
        self.weight_V = nn.Parameter(torch.zeros(self.weight_rank, in_features))
        nn.init.kaiming_uniform_(self.weight_U, a=math.sqrt(5))
        self.scaling = self.alpha / self.weight_rank

    def forward(self, inputs):
        self.mask_scores = self.weight_U @ self.weight_V * self.scaling
        # self.mask = F.sigmoid(200 * self.mask_scores) * self.scale
        self.new_weight = self.weight + self.mask_scores

        return F.linear(inputs, self.new_weight, self.bias)


class LoRaEmbedding(nn.Embedding):
    """
    Fully Connected layer with on the fly adaptive mask.
    """

    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            weight_rank=4,
            alpha=8
    ):
        super().__init__(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        self.weight_rank = weight_rank
        self.alpha = alpha

        # create trainable params
        self.weight_U = nn.Parameter(torch.zeros(num_embeddings, self.weight_rank))
        self.weight_V = nn.Parameter(torch.zeros(self.weight_rank, embedding_dim))
        nn.init.normal_(self.weight_V)
        self.scaling = self.alpha / self.weight_rank

    def forward(self, inputs):
        self.mask_scores = self.weight_U @ self.weight_V * self.scaling
        # self.mask = F.sigmoid(200 * self.mask_scores) * self.scale
        self.new_weight = self.weight + self.mask_scores

        return F.embedding(
                inputs, self.new_weight, self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )