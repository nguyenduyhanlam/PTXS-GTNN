# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:17:14 2020

@author: User
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import unidecode
import string
import re
import random
import numpy as np
import html
from torch.autograd import Variable

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

SOS_token = 0 #Start Of Sequence
EOS_token = 1 #End Of Sequence
PAD_token = 2 # Used for padding short sentences

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
# remove all the accents
def remove_accent(utf8_str):
    return unidecode.unidecode(utf8_str)

contractions = {
"ain 't": "am not / are not",
"aren 't": "are not / am not",
"can 't": "cannot",
"can 't 've": "cannot have",
"'cause": "because",
"could 've": "could have",
"couldn 't": "could not",
"couldn 't 've": "could not have",
"didn 't": "did not",
"doesn 't": "does not",
"don 't": "do not",
"hadn 't": "had not",
"hadn 't 've": "had not have",
"hasn 't": "has not",
"haven 't": "have not",
"he 'd": "he had / he would",
"he 'd 've": "he would have",
"he 'll": "he shall / he will",
"he 'll 've": "he shall have / he will have",
"he 's": "he has / he is",
"how'd": "how did",
"how'd 'y": "how do you",
"how 'll": "how will",
"how 's": "how has / how is",
"i 'd": "I had / I would",
"i 'd 've": "I would have",
"i 'll": "I shall / I will",
"i 'll 've": "I shall have / I will have",
"i 'm": "I am",
"i 've": "I have",
"isn 't": "is not",
"it 'd": "it had / it would",
"it 'd've": "it would have",
"it 'll": "it shall / it will",
"it 'll've": "it shall have / it will have",
"it 's": "it has / it is",
"let 's": "let us",
"ma 'am": "madam",
"mayn 't": "may not",
"might 've": "might have",
"mightn 't": "might not",
"mightn 't 've": "might not have",
"must 've": "must have",
"mustn 't": "must not",
"mustn 't 've": "must not have",
"needn 't": "need not",
"needn 't 've": "need not have",
"o 'clock": "of the clock",
"oughtn 't": "ought not",
"oughtn 't 've": "ought not have",
"shan 't": "shall not",
"sha 'n 't": "shall not",
"shan 't 've": "shall not have",
"she 'd": "she had / she would",
"she 'd've": "she would have",
"she 'll": "she shall / she will",
"she 'll've": "she shall have / she will have",
"she 's": "she has / she is",
"should 've": "should have",
"shouldn 't": "should not",
"shouldn 't 've": "should not have",
"so 've": "so have",
"so 's": "so as / so is",
"that 'd": "that would / that had",
"that 'd 've": "that would have",
"that 's": "that has / that is",
"there 'd": "there had / there would",
"there 'd 've": "there would have",
"there 's": "there has / there is",
"they 'd": "they had / they would",
"they 'd 've": "they would have",
"they 'll": "they shall / they will",
"they 'll 've": "they shall have / they will have",
"they 're": "they are",
"they 've": "they have",
"to 've": "to have",
"wasn 't": "was not",
"we 'd": "we had / we would",
"we 'd 've": "we would have",
"we 'll": "we will",
"we 'll 've": "we will have",
"we 're": "we are",
"we 've": "we have",
"weren 't": "were not",
"what 'll": "what shall / what will",
"what 'll 've": "what shall have / what will have",
"what 're": "what are",
"what 's": "what has / what is",
"what 've": "what have",
"when 's": "when has / when is",
"when 've": "when have",
"where 'd": "where did",
"where 's": "where has / where is",
"where 've": "where have",
"who 'll": "who shall / who will",
"who 'll 've": "who shall have / who will have",
"who 's": "who has / who is",
"who 've": "who have",
"why 's": "why has / why is",
"why 've": "why have",
"will 've": "will have",
"won 't": "will not",
"won 't 've": "will not have",
"would 've": "would have",
"wouldn 't": "would not",
"wouldn 't 've": "would not have",
"y 'all": "you all",
"y 'all 'd": "you all would",
"y 'all 'd 've": "you all would have",
"y 'all 're": "you all are",
"y 'all 've": "you all have",
"you 'd": "you had / you would",
"you 'd 've": "you would have",
"you 'll": "you shall / you will",
"you 'll 've": "you shall have / you will have",
"you 're": "you are",
"you 've": "you have"
}

# Normalize the string (marks and words are seperated, words don't contain accents,...)
def normalizeString(s):
    # Remove all the accents first.
    s = remove_accent(html.unescape(s))
    # Seperate words and marks by adding spaces between them
    marks = '[.!?,-${}()]'
    r = "(["+"\\".join(marks)+"])"
    s = re.sub(r, r" \1 ", s)
    # replace continuous spaces with a single space
    s = re.sub(r"\s+", r" ", s).strip()
    
    # Convert to writing form
    for c in contractions:
        if c in s:
            s = s.replace(c, contractions[c].lower())

    return s

# Example
ex_s = "Ăn quả, nhớ kẻ trồng cây."
ex_s1 = "I 'm your daddy"
print(normalizeString(ex_s)) # result will be "An qua , nho ke trong cay ."
print(normalizeString(ex_s1.lower()))

def readLangs(lang1, lang2):
    print("Reading lines...")

    # Read the file and split into lines
    lines_language1 = open('train.%s' % lang1, encoding='utf-8').\
        read().strip().split('\n')
    lines_language2 = open('train.%s' % lang2, encoding='utf-8').\
        read().strip().split('\n')

    # Normalize all the lines
    data_language1 = [normalizeString(l.lower().strip()) for l in lines_language1]
    data_language2 = [normalizeString(l.lower().strip()) for l in lines_language2]

    # Prepare return values
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    data = list(zip(data_language1, data_language2))

    return input_lang, output_lang, data

# Test the function
lang1 = "en"
lang2 = "vi"
input_lang, output_lang, data = readLangs(lang1, lang2)
print("Language 1:", input_lang.name)
print("Language 2:", output_lang.name)
print(random.choice(data))

MAX_LENGTH = 10

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def lemmatizer(sentence):
    for word in sentence.split():
        sentence = sentence.replace(word, wordnet_lemmatizer.lemmatize(word, pos="v"))
    return sentence

def prepareData(lang1, lang2):
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        # Lemmatization process
        inWord = lemmatizer(pair[0])
        outWord = lemmatizer(pair[1])
        # Add word to dictionaries after lemmatization
        input_lang.addSentence(inWord)
        output_lang.addSentence(outWord)
    print("Counted words:")
    print("Language 1:", input_lang.name, "There are", input_lang.n_words, "different words")
    print("Language 2:", output_lang.name, "There are", output_lang.n_words, "different words")
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('en', 'vi')
print(random.choice(pairs))

print(lemmatizer("I'm excited about watching that movie"))
print(lemmatizer("The pizza was delivered to the wrong address"))

class GRC(nn.Module):
    def __init__(self, char_enc_size, label_to_ind, rnn_type, emb_size, hidden_size, num_layers,
                  bidirectional,max_path, recurrent_drop=0, input_drop=0):
        super(GRC, self).__init__()
        self.char_embed = nn.Linear(char_enc_size, emb_size)
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size_seg = self.hidden_size
        self.recurrent_drop = recurrent_drop
        
        if self.bidirectional == True:
            self.NUM_DIRECTIONS = 2
        else:
            self.NUM_DIRECTIONS = 1
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(emb_size, self.hidden_size // self.NUM_DIRECTIONS, num_layers, batch_first=False,
                              dropout=self.recurrent_drop, bidirectional = self.bidirectional)
         
        self.WL = nn.Linear(self.hidden_size, self.hidden_size, bias=None)
        self.WR = nn.Linear(self.hidden_size, self.hidden_size, bias=None)
        
        self.bw = nn.Parameter(torch.FloatTensor(1, self.hidden_size).zero_())        
        
        self.GL = nn.Linear(self.hidden_size, 3*self.hidden_size, bias=None)
        self.GR = nn.Linear(self.hidden_size, 3*self.hidden_size, bias=None)
        
        self.bg = nn.Parameter(torch.FloatTensor(1, 3*self.hidden_size).zero_())     
        self.sigm = torch.nn.Sigmoid()    
        
        self.tag_to_ix = label_to_ind
        self.tagset_size = len(self.tag_to_ix)
        self.max_path = max_path
        
        self.drop3d = nn.Dropout3d(input_drop)
        self.drop = nn.Dropout(input_drop)
        
        self.fc = nn.Linear(self.hidden_size_seg,  self.tagset_size)
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
    
        self.init_weights()


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.num_layers*self.NUM_DIRECTIONS, batch_size, self.hidden_size // self.NUM_DIRECTIONS).zero_()),
                        Variable(weight.new(self.num_layers*self.NUM_DIRECTIONS, batch_size, self.hidden_size // self.NUM_DIRECTIONS).zero_()))
        else:
            return Variable(weight.new(self.num_layers*self.NUM_DIRECTIONS, batch_size, self.hidden_size // self.NUM_DIRECTIONS).zero_() )    
    
    
    def init_hidden_seg(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(1, batch_size, self.hidden_size_seg //2  ).zero_()),
                        Variable(weight.new(1, batch_size, self.hidden_size_seg //2).zero_()))
        else:
            return Variable(weight.new(1, batch_size, self.hidden_size_seg).zero_() ) 


    def forward(self, x_batch_s, hidden, sorted_vals):
        batch_size = x_batch_s.size(1)
        x_batch_s = self.drop3d(x_batch_s)
        emb = self.char_embed(x_batch_s)
        emb = self.drop(emb)
        
        pack_emb = torch.nn.utils.rnn.pack_padded_sequence(emb, sorted_vals)
        out, hidden = self.rnn(pack_emb, hidden)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out)
        
        max_len = unpacked.size(0)
        unpacked = unpacked.transpose(1,0)

        segment_feat = Variable(unpacked.data.new(batch_size, max_len, self.max_path, self.hidden_size).fill_(0))
        segment_feat[:, :, 0, :] = unpacked[:,:,:]
        
        left_feat = unpacked[:,:-1,:].contiguous().view(-1, self.hidden_size )
        right_feat = unpacked[:,1:,:].contiguous().view(-1, self.hidden_size )
        
        for i in range(1, self.max_path):
            next_level_cand = self.WL(left_feat) + self.WR(right_feat) + self.bw
            next_level_cand = 4*(self.sigm(next_level_cand) - 0.5)
            
            gate_feat = self.GL(left_feat) + self.GR(right_feat) + self.bg
            gate_feat = torch.exp(gate_feat)
            
            theta_scores =  tuple( gate_feat[:, i*self.hidden_size:(i+1)*self.hidden_size].unsqueeze(-1) for i in range(3) )
            theta_scores = torch.cat( theta_scores, 2)
            
            Z = torch.sum(theta_scores, 2)
            inv_z = 1/Z.unsqueeze(-1).expand(*Z.size(), 3)
            theta_norm = theta_scores * inv_z
            
            next_level_cand =   torch.mul(left_feat,theta_norm[:,:,0]) + torch.mul(right_feat,theta_norm[:,:,1]) + torch.mul(next_level_cand,theta_norm[:,:,2])
            next_level_cand = next_level_cand.view(batch_size, max_len-i, self.hidden_size)
            
            segment_feat[:, i:, i, :] = next_level_cand
            left_feat = next_level_cand[:,:-1,:].contiguous().view(-1, self.hidden_size )
            right_feat = next_level_cand[:,1:,:].contiguous().view(-1, self.hidden_size )

        #Get tag scores for crf
        segment_feat = self.fc(self.drop(segment_feat.view(-1, self.hidden_size) ))
        segment_feat = segment_feat.view(batch_size, max_len, self.max_path, self.tagset_size)
        
        return segment_feat, hidden    
  
    
    def init_weights(self):
        self.fc.bias.data.fill_(0)
        for name, param in self.named_parameters(): 
            if ('weight' in name): 
                print ('Initializing ', name) 
                initrange = np.sqrt( 6 / sum(param.size()))
                self.state_dict()[name].uniform_(-initrange, initrange)
             
        
    def _forward_alg(self, logits, len_list, is_volatile=False):
        """
        Computes the (batch_size,) denominator term (FloatTensor list) for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        
        Arguments:
            logits: [batch_size, seq_len, max_path, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, max_path, n_labels = logits.size()
        
        alpha = logits.data.new(batch_size, seq_len+1, self.tagset_size).fill_(-10000)
        alpha[:, 0, self.tag_to_ix['START']] = 0
        alpha = Variable(alpha, volatile=is_volatile)
        
        # Transpose batch size and time dimensions:
        logits_t = logits.permute(1,0,2,3)
        c_lens = len_list.clone()
        
        alpha_out_sum = Variable(logits.data.new(batch_size,max_path, self.tagset_size).fill_(0))
        mat = Variable(logits.data.new(batch_size,self.tagset_size,self.tagset_size).fill_(0))
        
        for j, logit in enumerate(logits_t):
            for i in range(0,max_path):
                if i<=j:
                    alpha_exp = alpha[:,j-i, :].clone().unsqueeze(1).expand(batch_size,self.tagset_size, self.tagset_size)
                    logit_exp = logit[:, i].unsqueeze(-1).expand(batch_size, self.tagset_size, self.tagset_size)
                    trans_exp = self.transitions.unsqueeze(0).expand_as(alpha_exp)
                    mat = alpha_exp + logit_exp + trans_exp
                    alpha_out_sum[:,i,:] =  log_sum_exp(mat , 2, keepdim=True)
                    
            alpha_nxt = log_sum_exp(alpha_out_sum , dim=1, keepdim=True).squeeze(1)
            
            mask = Variable((c_lens > 0).float().unsqueeze(-1).expand(batch_size,self.tagset_size))
            alpha_nxt = mask * alpha_nxt + (1 - mask) *alpha[:, j, :].clone() 
            
            c_lens = c_lens - 1      

            alpha[:,j+1, :] = alpha_nxt

        alpha[:,-1,:] = alpha[:,-1,:] + self.transitions[self.tag_to_ix['STOP']].unsqueeze(0).expand_as(alpha[:,-1,:])
        norm = log_sum_exp(alpha[:,-1,:], 1).squeeze(-1)

        return norm
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # A simple lookup table that stores embeddings of a fixed dictionary 
        # and size. 
        # This module is often used to store word embeddings and retrieve them using indices. 
        # The input to the module is a list of indices, 
        # and the output is the corresponding word embeddings.
        # input_size: num_embeddings (python:int) – size of the dictionary of embeddings
        # hidden_size: embedding_dim (python:int) – the size of each embedding vector
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
        # hidden_size: input_size – The number of expected features in the input x
        # hidden_size: hidden_size – The number of features in the hidden state h
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # Create word embedding table based on the data -> tensor table
        # view(*shape): Returns a new tensor with the same data as the self tensor but of a different shape.
        embedded = self.embedding(input).view(1, 1, -1)

        output = embedded

        # Execute GRU process
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        # Initiate the first hidden unit
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # A simple lookup table that stores embeddings of a fixed dictionary 
        # and size. 
        # This module is often used to store word embeddings and retrieve them using indices. 
        # The input to the module is a list of indices, 
        # and the output is the corresponding word embeddings.
        # output_size: num_embeddings (python:int) – size of the dictionary of embeddings
        # hidden_size: embedding_dim (python:int) – the size of each embedding vector
        self.embedding = nn.Embedding(output_size, hidden_size)

        # Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
        # hidden_size: input_size – The number of expected features in the input x
        # hidden_size: hidden_size – The number of features in the hidden state h
        self.gru = nn.GRU(hidden_size, hidden_size)

        # Applies a linear transformation to the incoming data: y = xA^T + b
        # hidden_size: in_features – size of each input sample
        # output_size: out_features – size of each output sample
        self.out = nn.Linear(hidden_size, output_size)
        
        # Applies the log(Softmax(x) function to an n-dimensional input Tensor. The LogSoftmax formulation can be simplified as:
        # logSoftmax(xi) = log(exp(xi) / sum_j(exp(xj)))
        # dim=1: Input: (*) where * means, any number of additional dimensions
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # Create word embedding table based on the data -> tensor table
        # view(*shape): Returns a new tensor with the same data as the self tensor but of a different shape.
        output = self.embedding(input).view(1, 1, -1)
        
        # Assuring the encoder output is not negative
        # Because ReLU output range is (0, +inf)
        output = F.relu(output)

        # Execute GRU process
        output, hidden = self.gru(output, hidden)

        # Calculate the softmax
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        # Initiate the first hidden unit
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # A simple lookup table that stores embeddings of a fixed dictionary 
        # and size. 
        # This module is often used to store word embeddings and retrieve them using indices. 
        # The input to the module is a list of indices, 
        # and the output is the corresponding word embeddings.
        # output_size: num_embeddings (python:int) – size of the dictionary of embeddings
        # hidden_size: embedding_dim (python:int) – the size of each embedding vector
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        # Applies a linear transformation to the incoming data: y = xA^T + b
        # hidden_size * 2: in_features – size of each input sample
        # max_length: out_features – size of each output sample
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        # Applies a linear transformation to the incoming data: y = xA^T + b
        # hidden_size * 2: in_features – size of each input sample
        # hidden_size: out_features – size of each output sample
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # During training, randomly zeroes some of the elements of 
        # the input tensor with probability p 
        # using samples from a Bernoulli distribution. 
        # Each channel will be zeroed out independently on every forward call.
        # This has proven to be an effective technique for regularization 
        # and preventing the co-adaptation of neurons 
        # as described in the paper Improving neural networks by preventing co-adaptation of feature detectors .
        # Furthermore, the outputs are scaled by a factor of 1/(1-p) 
        # during training. This means that during evaluation the module simply computes an identity function.
        # dropout_p: p – probability of an element to be zeroed. Default (when don't have parameter): 0.5
        # In this constructor, the default parameter (probability) is 0.1
        self.dropout = nn.Dropout(self.dropout_p)

        # Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
        # hidden_size: input_size – The number of expected features in the input x
        # hidden_size: hidden_size – The number of features in the hidden state h
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        
        # Applies a linear transformation to the incoming data: y = xA^T + b
        # hidden_size: in_features – size of each input sample
        # output_size: out_features – size of each output sample
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # Create word embedding table based on the data -> tensor table
        # view(*shape): Returns a new tensor with the same data as the self tensor but of a different shape.
        embedded = self.embedding(input).view(1, 1, -1)
        
        # Execute dropout process
        embedded = self.dropout(embedded)

        # Execute attention process
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # Performs a batch matrix-matrix product
        # unsqueeze: Returns a new tensor with a dimension of size one inserted at the specified position.
        #            The returned tensor shares the same underlying data with this tensor.
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        # Assuring the encoder output is not negative
        # Because ReLU output range is (0, +inf)
        output = F.relu(output)

        # Execute GRU process
        output, hidden = self.gru(output, hidden)

        # Calculate the softmax
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        # Initiate the first hidden unit
        return torch.zeros(1, 1, self.hidden_size, device=device)

def indexesFromSentence(lang, sentence):
    # Get the counter for each word in the sentence
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    # Get the counter
    indexes = indexesFromSentence(lang, sentence)
    # Append token to the end of the array
    indexes.append(EOS_token)
    # Return the vector
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    # Create input vector
    input_tensor = tensorFromSentence(input_lang, lemmatizer(pair[0]))
    # Create output vector
    target_tensor = tensorFromSentence(output_lang, lemmatizer(pair[1]))
    # Return result
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5

# Training function
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    
    # Intiatie encoder process
    encoder_hidden = encoder.initHidden()

    # Clears the gradients of all optimized torch.Tensor.
    encoder_optimizer.zero_grad()
    # Clears the gradients of all optimized torch.Tensor.
    decoder_optimizer.zero_grad()

    # Save input tensor size
    input_length = input_tensor.size(0)
    # Save output tensor size
    target_length = target_tensor.size(0)

    # Initiate encoder output
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    # Initiate loss value
    loss = 0

    # for each encoder input
    for ei in range(input_length):
        # Execute encoder process
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        # Save encoder result
        encoder_outputs[ei] = encoder_output[0, 0]

    # Initiate decoder input
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # Assign encoder result to the 1st decoder hidden unit
    decoder_hidden = encoder_hidden

    # Randomly using teacher_forcing feature
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            # Execute decoder process
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # Add loss
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            # Execute decoder process
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            # Add loss
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    # Backpropagation
    loss.backward()

    # Performs a single optimization step.
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

import tkinter
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # Implements stochastic gradient descent (optionally with momentum).
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # Randomly choose some pairs in the dataset for training process
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    # The negative log likelihood loss.
    # The input given through 
    # a forward call is expected to contain log-probabilities of each class. 
    # Input has to be a Tensor of size either (minibatch,C)
    # or or (minibatch, C, d1, d2, ..., dK) with K≥1 for the K-dimensional case.
    # Default: loss = sum((1/sum(w_(y_n))) * loss_n), loss_n = -w*x_(n,y_n)
    criterion = nn.NLLLoss()

    # for each sample
    for iter in range(1, n_iters + 1):
        # Get input data and target data
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        
        # Execute training process
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        # Print info
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    # Draw plot
    showPlot(plot_losses)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    # Disable the gradient computation.
    with torch.no_grad():
        # Get input sample
        input_tensor = tensorFromSentence(input_lang, lemmatizer(sentence))
        input_length = input_tensor.size()[0]

        # Initiate encoder process
        encoder_hidden = encoder.initHidden()

        # Initiate encoder output
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        # for each encoder input
        for ei in range(input_length):
            # Execute encoder process
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            # Save encoder result
            encoder_outputs[ei] += encoder_output[0, 0]

        # Initiate decoder input
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        # Assign encoder result to the 1st decoder hidden unit
        decoder_hidden = encoder_hidden

        # Initiate decoder output
        decoded_words = []
        # Initiate decoder attention
        decoder_attentions = torch.zeros(max_length, max_length)

        # for each decoder input
        for di in range(max_length):
            # Execute decoder process
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            # Save decoder attention
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
    #for pair in pairs:
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        
hidden_size = 2048
#hidden_size = 512
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 130000, print_every=5000)
#trainIters(encoder1, attn_decoder1, 75000, print_every=5000)

evaluateRandomly(encoder1, attn_decoder1)

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)

evaluateAndShowAttention(normalizeString("i know all the possibility .".lower().strip()))

evaluateAndShowAttention(normalizeString("they kill me .".lower().strip()))

evaluateAndShowAttention(normalizeString("let me tell you a story .".lower().strip()))