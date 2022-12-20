import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    '''
    Implements MHSA using the PyTorch MultiheadAttention Layer.
    '''
    def __init__(self, hidden_size, num_heads, dropout):
        '''
        Arguments:
            hidden_size: Dimension of the output of the self-attention.
            num_heads: Number of heads for the multi-head attention. 
            dropout: Dropout probability for the self-attention. 
        '''
        super().__init__()
        if hidden_size % num_heads != 0:
            print('The hidden size {} is not a multiple of the number of heads {}'.format(hidden_size, num_heads))
        self.attention_layer = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
    def forward(self, x, key_padding_mask=None, attention_mask=None):
        return self.attention_layer(query=x, key=x, value=x, key_padding_mask=key_padding_mask, attn_mask=attention_mask)

 
class FeedForward(nn.Module):
    '''
    Implements the feed-forward component of the transfomer model.
    '''
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        '''
        Arguments:
            input_dim: 
            hidden_dim: Hidden size of the Transformer that this feed-forward layer is part of.
            dropout: Dropout probability to use for the projected activations. If `0.0` then no dropout will be used.
        '''
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        x = self.layer_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        return x


class TransformerLayerNorm(nn.Module):
    '''
    Implements LayerNorm for self-attention and feed-forward networks.
    '''
    def __init__(self, hidden_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        x = x.to(self.layer_norm.weight.dtype)
        return self.layer_norm(x) 


class TransformerLayer(nn.Module):
    '''
    '''
    def __init__(self, embed_dim, num_heads, hidden_dim, attn_dropout=0.0, ffn_dropout=0.0):
        super().__init__()
        self.attn_layer = MultiHeadSelfAttention(embed_dim, num_heads, dropout=attn_dropout)
        self.ffn_layer = FeedForward(embed_dim, hidden_dim, dropout=ffn_dropout)
        self.layer_norm = TransformerLayerNorm(embed_dim)
    def forward(self, x, key_padding_mask=None, attention_mask=None):
        attn_out, attn_weights = self.attn_layer(x, key_padding_mask=None, attention_mask=None)
        x = self.layer_norm(x + attn_out)
        ffn_out = self.ffn_layer(x)
        x = self.layer_norm(x + ffn_out)
        return x, attn_weights
     
    
 class TransformerEncoder(nn.Module):
    '''
    '''
    def __init__(self, num_layers, num_heads, embed_dim, hidden_dim, attn_dropout=0.0, ffn_dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=embed_dim, attn_dropout=attn_dropout, ffn_dropout=ffn_dropout) for _ in range(num_layers)])
        self.attn_weights = []
    def forward(self, x, key_padding_mask=None, attention_mask=None):
        for layer in self.layers:
            x, weights = layer(x, key_padding_mask=None, attention_mask=None)
            self.attn_weights.append(weights)
        return x
    def get_attention_weights(self):
        if len(self.attn_weights) != 0:
            return self.attn_weights
        else:
            print("The model hasn't been training yet")
            
            
  class PositionalEncoding(nn.Module):
    '''
    '''
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0)/embed_dim))
        self.pe = torch.zeros(max_len, 1, embed_dim)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
  
 
class MyGPT(nn.Module):
    '''
    '''
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, hidden_dim, ffn_droput, attn_dropout, num_class):
        super(MyGPT, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.transformer_encoder = TransformerEncoder(num_layers, num_heads, embed_dim, hidden_dim=hidden_dim, attn_dropout=attn_dropout, ffn_dropout=ffn_droput)
        self.decoder = nn.Linear(embed_dim, num_class)
        self.embed_layer_norm = nn.LayerNorm(embed_dim)
        self.init_weights()

    def init_weights(self):
        init_range = 0.5
        self.decoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        
    
    def forward(self, seqs, attn_mask=None, key_padding_mask=None):
        embedded_seq = self.embedding_layer(seqs)
        embedded_seq = self.pos_encoder(embedded_seq)
        embedded_seq = self.embed_layer_norm(embedded_seq)
        out = self.transformer_encoder(x=embedded_seq, key_padding_mask=key_padding_mask, attention_mask=attn_mask)
        results = self.decoder(out)
        return results
