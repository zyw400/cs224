#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    The CNN module in the embedding
    """

    def __init__(self, e_char, e_word, m_word = 14, k = 5, padding = 1):
        """
        Init the layers for the network
        @param e_char (int): input size
        @param e_word (int): filter number, set to e_word
        @param k (int): kernel size
        """
        super(CNN, self).__init__()
        self.projection = nn.Conv1d(in_channels=e_char, out_channels=e_word,
                                    kernel_size=k, padding = padding)

        # self.maxpool = nn.MaxPool1d(kernel_size=m_word - k + 1)
        # self.k = k

    def forward(self, x_reshape):
        """
        Calculate CNN layer outputs.
        @param x_reshape: Tensor of shape (batch_size, e_char, m_word)

        @param x_conv_out: Tensor of shape (batch_size, e_word)
        """
        # print("x_reshape {}".format(x_reshape.size()))
        # print(x_reshape.size())
        x_conv = self.projection(x_reshape)
        # print("x_conv {}".format(x_conv.size()))
        # x_conv_out = self.maxpool(F.relu(x_conv)).squeeze(-1)
        # print("x_conv_out: {}".format(x_conv_out.size()))
        # x_conv_out = F.relu(x_conv).max(dim = -1).values.squeeze(-1)
        x_conv_out = torch.max(F.relu(x_conv), dim=2)[0]
       
        return x_conv_out