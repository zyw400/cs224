#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size), length is the total number of chars
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.
        
        # print(input.size())

        x = self.decoderCharEmb(input) #(seq_length, batch_size,char_embedding_size)

        dec_hidden, (last_hidden, last_cell) = self.charDecoder(x, dec_hidden) #(1, batch_size,hidden_size)
        
        # print(last_hidden.size())
        scores = self.char_output_projection(dec_hidden)
        return scores, (last_hidden, last_cell)

        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        
        seq_length = char_sequence.size()[0]
        batch_size = char_sequence.size()[1]

        vocab_size = len(self.target_vocab.char2id)

        scores, dec_hidden = self.forward(char_sequence[:-1], dec_hidden = dec_hidden)

        loss = nn.CrossEntropyLoss(ignore_index = self.target_vocab.char_pad, reduction='sum') # learned from somewhere else

        return loss(scores.reshape((seq_length - 1) * batch_size, vocab_size), char_sequence[1:].reshape((seq_length - 1) * batch_size ))


        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        output_word = []
        batch_size = initialStates[0].size()[1]

        # aways assume sequence length is 1, and start from the previous state
        current_char = torch.ones(1, batch_size, dtype=torch.long,
            device = device) * self.target_vocab.start_of_word
        state = initialStates
        # print(current_char.size())
        if_no_ended = torch.ones(batch_size)

        for t in range(max_length):

            scores, state = self.forward(current_char, state)
            # print(scores.size())
           
            current_char = torch.argmax(scores, dim = -1).long()
            # print(current_char.size())

            if_no_ended = (1 - (current_char[0] == self.target_vocab.end_of_word).float()) * if_no_ended

            output_word.append([self.target_vocab.id2char[int(current_char[0][i])] if if_no_ended[i] > 0 else "" for i in range(batch_size)])
        
        decodedWords = ["".join(x) for x in list(map(list, zip(*output_word)))]

        return decodedWords


        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        ### END YOUR CODE

