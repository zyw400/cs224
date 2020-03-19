# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# CS224N 2019-20: Homework 5
# model_embeddings.py: Embeddings for the NMT model
# Pencheng Yin <pcyin@cs.cmu.edu>
# Sahil Chopra <schopra8@stanford.edu>
# Anand Dhoot <anandd@stanford.edu>
# Michael Hahn <mhahn2@stanford.edu>
# """

# import torch.nn as nn
# import torch
# # Do not change these imports; your module names should be
# #   `CNN` in the file `cnn.py`
# #   `Highway` in the file `highway.py`
# # Uncomment the following two imports once you're ready to run part 1(j)

# from cnn import CNN
# from highway import Highway


# # End "do not change"

# class ModelEmbeddings(nn.Module):
#     """
#     Class that converts input words to their CNN-based embeddings.
#     """

#     def __init__(self, word_embed_size, vocab, char_embed_size = 50, dropout_rate = 0.3):
#         """
#         Init the Embedding layer for one language
#         @param word_embed_size (int): Embedding size (dimensionality) for the output word
#         @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

#         Hints: - You may find len(self.vocab.char2id) useful when create the embedding
#         """
#         super(ModelEmbeddings, self).__init__()

#         self.embedding = nn.Embedding(len(vocab.char2id), char_embed_size, padding_idx=vocab.char_pad)
#         self.cnn = CNN(char_embed_size, word_embed_size)
#         self.highway = Highway(word_embed_size)
#         self.dropout = nn.Dropout(p = dropout_rate)
#         self.word_embed_size = word_embed_size
#         self.char_embed_size = char_embed_size
#         print("word_embeded_size {}".format(word_embed_size))

#         ### YOUR CODE HERE for part 1h

#         ### END YOUR CODE

#     def forward(self, input):
#         """
#         Looks up character-based CNN embeddings for the words in a batch of sentences.
#         @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
#             each integer is an index into the character vocabulary

#         @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
#             CNN-based embeddings for each word of the sentences in the batch
#         """
#         # x_emb = self.embedding(input)
#         # x_reshaped = x_emb.permute(0,1,3,2)

#         # dim_0 = x_reshaped.size()[0]
#         # dim_1 = x_reshaped.size()[1]

#         # x_reshaped_flattened = torch.flatten(x_reshaped, start_dim=0, end_dim=1)

#         # x_conv_out = self.cnn(x_reshaped_flattened)
#         # x_highway = self.highway(x_conv_out)
#         # x_word_emb = self.dropout(x_highway)

#         # x_word_emb_unflatten = x_word_emb.reshape(dim_0,dim_1,x_highway.size()[-1])       


#         X_word_emb_list = []
#         # print("input size: {}".format(input.size()))
#         # divide input into sentence_length batchs
#         for X_padded in input:
#             # print("X_padded {}".format(X_padded.size()))

#             X_emb = self.embedding(X_padded)
#             X_reshaped = torch.transpose(X_emb, dim0=-1, dim1=-2)
#             # conv1d can only take 3-dim mat as input
#             # so it needs to concat/stack all the embeddings of word
#             # after going through the network
#             # print("X_shaped {}".format(X_reshaped.size()))
#             X_conv_out = self.cnn(X_reshaped)

#             X_highway = self.highway(X_conv_out)
#             X_word_emb = self.dropout(X_highway)
#             X_word_emb_list.append(X_word_emb)

#         x_word_emb_unflatten = torch.stack(X_word_emb_list)
#         # return x_word_emb_unflatten
#         ### YOUR CODE HERE for part 1j
#         # print("input: ")
#         # print(input.size())
#         batch_size, seq_len, max_word_length = input.shape[1], input.shape[0], input.shape[2]
#         # print('batch size', batch_size)
#         # print('max word length', max_word_length)
#         #print('seq len', seq_len)

#         x_char_embed = self.embedding(input)  # shape: (sentence_length, batch_size, max_word_length, e_char)
#         #print('x_char embed shape', x_char_embed.shape)
#         x_reshaped = x_char_embed.permute(0, 1, 3, 2)  # shape: (sentence_length, batch_size, e_char, max_word_length)
#         #print('x_reshaped shape', x_reshaped.shape)
#         x_conv = self.cnn(x_reshaped.view(-1, self.char_embed_size, max_word_length)) # shape (seq_len*batch_size, e_word)
#         #print('x_conv shape', x_conv.shape)
#         x_highway = self.highway(x_conv)  # shape: (batch_size*seq_len, e_word)
#         #print('x_highway shape', x_highway.shape)
#         x_word_embed = self.dropout(x_highway.view(seq_len, batch_size, self.word_embed_size))
        

#         return x_word_embed

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway
from vocab import VocabEntry

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab: VocabEntry):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        #pad_token_idx = vocab.src['<pad>']
        #self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.char_embed_size = 50
        self.word_embed_size = embed_size
        self.char_embedding = nn.Embedding(len(vocab.char2id), self.char_embed_size, padding_idx=vocab.char_pad)
        self.cnn = CNN(e_char=self.char_embed_size, e_word=embed_size)
        self.highway = Highway(word_embed_size=embed_size)
        self.dropout = nn.Dropout(0.3)
        ### END YOUR CODE

    def forward(self, input: torch.Tensor):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        batch_size, seq_len, max_word_length = input.shape[1], input.shape[0], input.shape[2]
        # print('batch size', batch_size)
        # print('max word length', max_word_length)
        #print('seq len', seq_len)

        x_char_embed = self.char_embedding(input)  # shape: (sentence_length, batch_size, max_word_length, e_char)
        #print('x_char embed shape', x_char_embed.shape)
        x_reshaped = x_char_embed.permute(0, 1, 3, 2)  # shape: (sentence_length, batch_size, e_char, max_word_length)
        #print('x_reshaped shape', x_reshaped.shape)
        x_conv = self.cnn(x_reshaped.view(-1, self.char_embed_size, max_word_length)) # shape (seq_len*batch_size, e_word)
        #print('x_conv shape', x_conv.shape)
        x_highway = self.highway(x_conv)  # shape: (batch_size*seq_len, e_word)
        #print('x_highway shape', x_highway.shape)
        x_word_embed = self.dropout(x_highway.view(seq_len, batch_size, self.word_embed_size))
        #print('x_word embed shape', x_word_embed.shape)

        return x_word_embed
        ### END YOUR CODE

