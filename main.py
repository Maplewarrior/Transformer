import torch
from model import Transformer
from init_model_weights import _initializeModel

d_model = 1024 # Dimension of embeddings
n_heads = 8 # Number of heads for MHA
d_k = d_model/n_heads # dimension of keys (d_model / n_heads)
d_ff = d_model*4 # DON'T CHANGE!!! (be careful)

n_layers = 6 # Number of model layers
epochs = 3
pre_vocab_size = 64139
model = Transformer(pre_vocab_size, pre_vocab_size, d_model, d_ff, d_k, n_layers, n_heads)

lr = 0.00001 # 0.0001 default in "AIAYN"
optim = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)

