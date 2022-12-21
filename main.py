import torch
from gpt import MyGPT

vocab_size = 10
seq_length = 8
num_seqs = 5
x = torch.randint(vocab_size, (num_seqs, seq_length))
print(x.shape)

model = MyGPT(vocab_size, embed_dim=64, num_layers=2, num_heads=2, attn_dropout=0.3, ffn_dropout=0.1)
y = model(x)
print(y)
print(y.shape)
