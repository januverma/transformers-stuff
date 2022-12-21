import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    '''
    Implements MHSA using the PyTorch MultiheadAttention Layer.
    '''
    def __init__(self, hidden_dim, num_heads, dropout):
        '''
        Arguments:
            hidden_dim: Dimension of the output of the self-attention.
            num_heads: Number of heads for the multi-head attention. 
            dropout: Dropout probability for the self-attention. If `0.0` then no dropout will be used.
            
        Returns:
            A tensor of shape `num_tokens x hidden_size` containing output of the MHSA for each token.
        '''
        super().__init__()
        if hidden_dim % num_heads != 0:
            print('The hidden size {} is not a multiple of the number of heads {}'.format(hidden_dim, num_heads))
        self.attention_layer = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
    def forward(self, x, key_padding_mask=None, attention_mask=None):
        '''
        Arguments:
            x: Tensor containing input token embeddings.
            key_padding_mask: Mask indicating which elements within the input sequence to be considered as padding and ignored for the computation of self-attention scores.  
            attention_mask: Mask indicating which relative positions are allowed to attend.  
        '''
        return self.attention_layer(query=x, key=x, value=x, key_padding_mask=key_padding_mask, attn_mask=attention_mask)

 
class FeedForward(nn.Module):
    '''
    Implements the feed-forward component of the transfomer model.
    '''
    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        '''
        Arguments:
            input_dim: Dimension of the token embedding, output of the MHSA layer.
            hidden_dim: Hidden size of the Transformer that this feed-forward layer is part of.
            dropout: Dropout probability to use for the projected activations. If `0.0` then no dropout will be used.
        Returns:
            A tensor of shape `num_tokens x hidden_dim` containing projections for each token.
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

    Arguments:
        input_dim: Input dimension.
    
    Returns:
        A normalized tensor of the same dimension as the input. 
    '''
    def __init__(self, input_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
    def forward(self, x):
        x = x.to(self.layer_norm.weight.dtype)
        return self.layer_norm(x) 


class TransformerLayer(nn.Module):
    '''
    A transformer layer which is a sequential model consisting of self-attention, layer norm, residual connection, feed-forward projection, layer norm, residual connection. 
    
    Arguments:
        hidden_dim: Hidden dimension transformer layers.  
        num_heads: Number of attention heads. 
        attn_dropout: Dropout for MHSA layers. 
        ffn_dropout: Dropout for feed-forward layers.
    Returns:
        A tensor containing attention scores for each token. 
        attn_weights: A tensor of shape `num_tokens x num_tokens` containing the attention weights. 
    '''
    def __init__(self, hidden_dim, num_heads, attn_dropout=0.0, ffn_dropout=0.0):
        super().__init__()
        self.attn_layer = MultiHeadSelfAttention(hidden_dim, num_heads, dropout=attn_dropout)
        self.ffn_layer = FeedForward(hidden_dim, hidden_dim, dropout=ffn_dropout)
        self.layer_norm = TransformerLayerNorm(hidden_dim)
    def forward(self, x, key_padding_mask=None, attention_mask=None):
        attn_out, attn_weights = self.attn_layer(x, key_padding_mask, attention_mask)
        x = self.layer_norm(x + attn_out)
        ffn_out = self.ffn_layer(x)
        x = self.layer_norm(x + ffn_out)
        return x, attn_weights
     
    
class TransformerEncoder(nn.Module):
    '''
    Transformer Encoder which is composed for a stack of TransformerLayers. 

    Arguments:
        num_layers: Number of Transformer layers in the encoder. 
        hidden_dim: Hidden dimension of the transformer layers.  
        num_heads: Number of heads. 
        attn_dropout: Dropout for MHSA layers. 
        ffn_dropout: Dropout for feed-forward layers.
    '''
    def __init__(self, num_layers, hidden_dim, num_heads, attn_dropout=0.0, ffn_dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(hidden_dim, num_heads, attn_dropout, ffn_dropout) for _ in range(num_layers)])
        self.attn_weights = []
    def forward(self, x, key_padding_mask=None, attention_mask=None):
        for layer in self.layers:
            x, weights = layer(x, key_padding_mask, attention_mask)
            self.attn_weights.append(weights)
        return x
    def get_attention_weights(self):
        if len(self.attn_weights) != 0:
            return self.attn_weights
        else:
            print("The model hasn't been training yet")
            
            
class PositionalEncoding(nn.Module):
    '''
    Implements the sinusoidal positional encoding for the input tokens. 

    Arguments:
        embed_dim: Dimension of the positional encoding, should be the same as input token embedding. 
        dropout: Dropout probability to be used for positional encoding. 
        max_len: Maximum length of the input token sequences. 
    
    Returns:
      A tensor containing positional embeddings for each token.
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
    GPT-like language model.

    Arguments:
        vocab_size: Size of vocabulary.
        embed_dim: Dimension of the input token embedding. 
        num_layers: Number of Transformer layers in the encoder. 
        hidden_dim: Hidden dimension of the transformer layers.  
        ffn_hidden_dim: Hidden dimension of the Feed-forward layers. 
        num_heads: Number of heads. 
        attn_dropout: Dropout for MHSA layers. 
        ffn_dropout: Dropout for feed-forward layers.
    '''
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, attn_dropout, ffn_dropout):
        super(MyGPT, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.transformer_encoder = TransformerEncoder(num_layers, embed_dim, num_heads, attn_dropout, ffn_dropout)
        self.decoder = nn.Linear(embed_dim, vocab_size)
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
