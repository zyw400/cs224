#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Highway(nn.Module):
#     """
#     The Highway Network as in https://arxiv.org/abs/1505.00387
#     """

#     def __init__(self, e_word):
#         """
#         Init the layers for the network
#         @param e_word (int): input size
#         """
#         super(Highway, self).__init__()
#         self.h_projection = nn.Linear(e_word, e_word, bias = True)
#         self.h_gate = nn.Linear(e_word, e_word, bias = True)

#     def forward(self, x_conv_out):
#         """
#         Looks up character-based CNN embeddings for the words in a batch of sentences.
#         @param x_conv_out: Tensor of shape (batch_size, e_word)

#         @param x_highway: Tensor of shape (batch_size, e_word)
#         """

#         x_proj = F.relu(self.h_projection(x_conv_out))
#         x_gate = torch.sigmoid(self.h_gate(x_conv_out))

#         x_highway = x_gate * x_proj + (1.0 - x_gate) * x_conv_out

#         return x_highway
import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, word_embed_size):
        super().__init__()
        self.proj_layer = nn.Linear(word_embed_size, word_embed_size, bias=True)
        self.gate_layer = nn.Linear(word_embed_size, word_embed_size, bias=True)

    def forward(self, input: torch.Tensor):
        """
        :param input: tensor with the shape (batch_size, e_word)
        :return:
        """
        x_proj = F.relu(self.proj_layer(input))
        x_gate = torch.sigmoid(self.gate_layer(input))
        x_highway = x_gate * x_proj + (1 - x_gate) * input
        return x_highway

### END YOUR CODE 

