import numpy as np
import torch
import torch.nn as nn
import copy


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        
        self.d_model = d_model
        
        # Allocate memory to 
        pe = torch.zeros((max_seq_len, d_model))
        
        ### From attention is all you need ###
        for pos in range(max_seq_len):
            for i in range(0,d_model,2):
                pe[pos, i] = np.sin(pos/10000**(2*i/self.d_model))
                pe[pos, i+1] = np.cos(pos/10000**(2*i/self.d_model))
        # Fixed positional encoding
        pe.requires_grad = False
        #pe = pe.unsqueeze(0) # Make pe into [batch size x seq_len x d_model]
        self.register_buffer('pe',pe)
        
    def forward(self,x):

        # Make embeddings larger
        x = x*np.sqrt(self.d_model)
        # Get sequence length
        seq_len = x.size(1)
        
        pe = self.pe.clone()
        pe = pe.unsqueeze(0)
        
        x = x + torch.autograd.Variable(pe[:,:seq_len], 
        requires_grad=False)
        return x


def Attention(Q, K, V, d_k, mask=None, dropout=None):

    vals = (Q @ K.transpose(-2,-1))/np.sqrt(d_k)
    
    # Mask the scores if mask is specified. Model cannot see into future if masked.
    if mask is not None:
        mask = mask.unsqueeze(1)
        vals = vals.masked_fill(mask, 1e-9)
    # vals = vals if mask is None else vals.masked_fill_(mask, 1e-4)
    
    softmax = nn.Softmax(dim=-1)
    vals = softmax(vals)
    
    # apply dropout if specified
    vals = vals if dropout is None else dropout(vals)
    
    return vals @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, dropout=.1):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        # self.seq_len = seq_len
        self.d_k = d_k
        
        self.linearQ = nn.Linear(d_model, d_model)
        self.linearK = nn.Linear(d_model, d_model)
        self.linearV = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    
    # d_model = 512
    # n_heads = 8
    # seq_len = 20
    
    # [20,512] --> [20, 8, 64]
    ## If batch size is used, say of 128:
    ## out = [128, 20, 8, 64]
    
    # Input = Matrix of dim [bs x seq_len x d_model]
    def split_heads(self, t):
        return t.reshape(t.size(0), -1, self.n_heads, int(self.d_k))
    # Output = Matrix of dim [bs x seq_len x n_heads x d_k]
    
    def forward(self, Q, K, V, mask = None):
        
        Q = self.linearQ(Q)
        K = self.linearK(K)
        V = self.linearV(V)
        
        Q, K, V = [self.split_heads(t) for t in (Q, K, V)] 
        Q, K, V = [t.transpose(1,2) for t in (Q, K, V)] # reshape to [bs x n_heads x seq_len x d_k]
        
        # Compute Attention
        vals = Attention(Q, K, V, self.d_k, mask, self.dropout, self.align)
        
        # Reshape to [bs x seq_len x d_model]
        vals = vals.transpose(1,2).contiguous().view(vals.size(0), -1, self.d_model)
       
        out = self.out(vals) # linear
        
        
        return out
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=.1):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):

        return self.linear2(self.dropout(F.relu(self.linear1(x))))
    

        
class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_k, dropout=.1):
        
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, d_model, d_k, dropout)
        self.ffns = FeedForwardNetwork(d_model, d_ff, dropout)
        
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # See "Attention is all you need" to follow code structure
        x2 = self.dropout1(self.attention(x, x, x, mask))
        x = self.layer_norm1(x) + self.layer_norm1(x2)
        
        x2 = self.dropout2(self.ffns(x))
        x = x + self.layer_norm2(x2)
    
        return x
    

class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_k, dropout=.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.attention = MultiHeadAttention(n_heads, d_model, d_k, dropout)
        self.ffns = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # Batch Normalization
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # self.linear = nn.Linear()
        
    def forward(self, x, e_out, source_mask, target_mask):
        
        # See "Attention is all you need" to follow code structure
        ## part 1
        x2 = self.layer_norm1(x) # Norm
        x = self.dropout1(self.attention.forward(x2, x2, x2, target_mask)) # Masked MHA, target
        x = x2 + self.layer_norm1(x) # Add & Norm
        
        ## part 2
        x3 = self.dropout2(self.attention.forward(x, e_out, e_out, source_mask)) # MHA on encoder output
        x2 = self.dropout2(self.attention.forward(x, x, x)) #MHA continued in decoder
        x = self.layer_norm1(x3) + self.layer_norm1(x2) + self.layer_norm1(x) # Add & Norm
        
        ## part 3
        x2 = self.dropout3(self.ffns.forward(x)) ## Feed forward
        x = x + self.layer_norm2(x2) # add
        # x = self.norm3(x) # norm (!!!CHECK IF THIS IS EQUIVALENT!!!)
        return x

def cloneLayers(module, n_layers):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n_layers)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, d_k, n_layers, n_heads, dropout=.1):
        super().__init__()
        self.n_layers = n_layers
        self.embedder = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.e_layers = cloneLayers(EncoderLayer(n_heads, d_model, d_ff, d_k), n_layers)
        
    def forward(self, source, mask=None):
        x = self.embedder.forward(source)
        x = self.pe.forward(x)
        for i in range(self.n_layers):
            x = self.e_layers[i](x, mask)
        
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, d_k, n_layers, n_heads, dropout=.1):
        super().__init__()
        self.n_layers = n_layers
        self.embedder = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.d_layers = cloneLayers(DecoderLayer(n_heads, d_model, d_ff, d_k), n_layers)
        
    
    def forward(self, trg, e_out, source_mask, target_mask):
        x = self.embedder.forward(trg)
        x = self.pe.forward(x)
        
        for i in range(self.n_layers):
            x = self.d_layers[i](x, e_out, source_mask, target_mask)
        
        return x
        
class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, d_model,d_ff, d_k, n_layers, n_heads):
        super().__init__()
        self.e = Encoder(source_vocab_size, d_model,d_ff, d_k, n_layers, n_heads)
        self.d = Decoder(target_vocab_size, d_model,d_ff, d_k, n_layers, n_heads)
        self.linear_f = nn.Linear(d_model, target_vocab_size)
        #self.attentionAL = MultiHeadAttention(n_heads=1, d_model=d_model, d_k=d_k, alignment=alignment)
        #self.alignLayer = AlignmentLayer(source_vocab_size, target_vocab_size, d_model, d_ff, d_k, n_layers, n_heads=1)

        
    def forward(self, source, target, source_mask, target_mask):
        e_out = self.e.forward(source, source_mask)
        d_out = self.d.forward(target, e_out, source_mask, target_mask)
        out = self.linear_f(d_out)
        return out