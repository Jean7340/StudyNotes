#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import numpy as np
import torch
import torch.nn as nn
 
#%%
 
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, n_heads, qkv_bias=False):
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"
        self.d_out = d_out
        self.n_heads = n_heads
        self.d_k = d_out // n_heads # Reduce the projection dim to match desired output dim
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
 
    def forward(self, x):
        b, seq_len, d_in = x.shape
        k = self.W_k(x) # Shape: (b, seq_len, d_out)
        q = self.W_q(x)
        v = self.W_v(x)
        # We implicitly split the matrix by adding a `n_heads` dimension
        # Unroll last dim: (b, seq_len, d_out) -> (b, seq_len, n_heads, d_k)
        k = k.view(b, seq_len, self.n_heads, self.d_k) 
        v = v.view(b, seq_len, self.n_heads, self.d_k)
        q = q.view(b, seq_len, self.n_heads, self.d_k)
        # (b, seq_len, n_heads, d_k) -> (b, n_heads, seq_len, d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # Compute scaled dot-product attention with a causal mask
        attn_scores = q @ k.transpose(2, 3) # dot product for each head
        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:seq_len, :seq_len]
        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)        
        attn_weights = torch.softmax(attn_scores / self.d_k**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # Shape: (b, seq_len, n_heads, d_k)
        context_vec = (attn_weights @ v).transpose(1, 2)
        # Combine heads, where self.d_out = self.n_heads * self.d_k
        context_vec = context_vec.contiguous().view(b, seq_len, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection
        return context_vec
 
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
 
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
 
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])   )
 
    def forward(self, x):
        return self.layers(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            n_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"] )
        self.ff = FeedForward( cfg )
        self.norm1 = LayerNorm( cfg["emb_dim"] )
        self.norm2 = LayerNorm( cfg["emb_dim"] )
        self.drop_shortcut = nn.Dropout( cfg["drop_rate"] )
 
    def forward(self, x):        
        shortcut = x # Shortcut connection for attention block
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back        
        shortcut = x # Shortcut connection for feed forward block
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        return x
    
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding( cfg["vocab_size"], cfg["emb_dim"] )
        self.pos_emb = nn.Embedding( cfg["context_length"], cfg["emb_dim"] )
        self.drop_emb = nn.Dropout( cfg["drop_rate"] )
        self.trf_blocks = nn.Sequential( *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])] )       
        self.final_norm = LayerNorm( cfg["emb_dim"] )
        self.out_head = nn.Linear( cfg["emb_dim"], cfg["vocab_size"], bias=False )
 
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
 
GPT2CONFIG = {"vocab_size": 50257, "context_length": 1024, "drop_rate": 0.0, "qkv_bias": True }
GPT2SIZE = { "gpt2-small": {"emb_dim": 768, "n_layers": 12, "n_heads": 12, "size": "124M"},  # 621.83 MB
             "gpt2-medium": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16, "size":"355M"}, # 1549.58 MB
             "gpt2-large": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20, "size":"774M"},  # 3197.56 MB
             "gpt2-xl": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25, "size":"1558M"} }   # 6247.68 MB
 
#%% utility functions for training and inference
 
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor
 
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())
 
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss
 
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
 
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens): 
        idx_cond = idx[:, -context_size:] # crop current context to at most context_size number of ids
        with torch.no_grad():
            logits = model(idx_cond) # (batch, n_tokens, vocab_size)
        logits = logits[:, -1, :] # (batch, vocab_size), with focus on the last time step.        
        if top_k is not None: # keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)
        if temperature > 0.0: # apply temperature scaling
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)  # (batch_size, vocab_size)            
            idx_next = torch.multinomial(probs, num_samples=1) # (batch_size, 1) after sampling
        else: # greedy selection of the idx of the vocab with the highest probability
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)
        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break
        idx = torch.cat((idx, idx_next), dim=1) # (batch_size, n_tokens+1) after appending id_next to the running sequence
    return idx
 
 
#%% utility functions for loading pretrained GPT2 models
 
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))
 
def load_pretrained(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split( (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1 )
        gpt.trf_blocks[b].att.W_q.weight = assign( gpt.trf_blocks[b].att.W_q.weight, q_w.T )
        gpt.trf_blocks[b].att.W_k.weight = assign( gpt.trf_blocks[b].att.W_k.weight, k_w.T )
        gpt.trf_blocks[b].att.W_v.weight = assign( gpt.trf_blocks[b].att.W_v.weight, v_w.T )
        # load bias
        q_b, k_b, v_b = np.split( (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1 )
        gpt.trf_blocks[b].att.W_q.bias = assign( gpt.trf_blocks[b].att.W_q.bias, q_b )
        gpt.trf_blocks[b].att.W_k.bias = assign( gpt.trf_blocks[b].att.W_k.bias, k_b )
        gpt.trf_blocks[b].att.W_v.bias = assign( gpt.trf_blocks[b].att.W_v.bias, v_b )
        # load weights and bias for the output projection layer of mha
        gpt.trf_blocks[b].att.out_proj.weight = assign( gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T )
        gpt.trf_blocks[b].att.out_proj.bias = assign(   gpt.trf_blocks[b].att.out_proj.bias,   params["blocks"][b]["attn"]["c_proj"]["b"] )
        # load weights and bias for feedforward layers
        gpt.trf_blocks[b].ff.layers[0].weight = assign( gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(   gpt.trf_blocks[b].ff.layers[0].bias,   params["blocks"][b]["mlp"]["c_fc"]["b"] )
        gpt.trf_blocks[b].ff.layers[2].weight = assign( gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(   gpt.trf_blocks[b].ff.layers[2].bias,   params["blocks"][b]["mlp"]["c_proj"]["b"] )
        # load parameters for scale and shift of LayerNorm layers
        gpt.trf_blocks[b].norm1.scale = assign( gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"] )
        gpt.trf_blocks[b].norm1.shift = assign( gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"] )
        gpt.trf_blocks[b].norm2.scale = assign( gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"] )
        gpt.trf_blocks[b].norm2.shift = assign( gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"] )
    gpt.final_norm.scale = assign( gpt.final_norm.scale, params["g"] )
    gpt.final_norm.shift = assign( gpt.final_norm.shift, params["b"] )
    # weight tying: the original GPT-2 model reused the token embedding weights in the output layer to reduce the total number of parameters.
    gpt.out_head.weight  = assign( gpt.out_head.weight, params['wte'] )
    
#%%
 
if __name__ == "__main__":
    
    choice = "gpt2-small"
    cfg = GPT2CONFIG.copy()
    cfg.update( GPT2SIZE[choice] )
    cfg['qkv_bias'] = False
    cfg['drop_rate'] = 0.1
    
    model = GPTModel( cfg )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
 
    print("Token embedding layer shape:", model.tok_emb.weight.shape) # nn.Embedding(num_embeddings, embedding_dim) has weight of shape (num_embeddings, embedding_dim)
    print("Output layer shape:", model.out_head.weight.shape) # nn.Linear(in_features, out_features) has weight of shape (out_features,in_features)
 
    total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")
 
    total_size_bytes = total_params * 4 # total size in bytes (assuming float32, 4 bytes per parameter)
    print(f"Total size of the model: {total_size_bytes / (1024 * 1024):.2f} MB")
    
#%%
    torch.manual_seed(123)
 
    x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
    mha = MultiHeadAttention(d_in=768, d_out=768, context_length=1024, dropout=0.0, n_heads=12)
    print("context_vecs.shape:", mha(x).shape )
 
    block = TransformerBlock( cfg )
    print("Input shape:", x.shape)
    print("Output shape:", block(x).shape)
 
#%%
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append( torch.tensor(tokenizer.encode(txt1)) )
    batch.append( torch.tensor(tokenizer.encode(txt2)) )
    batch = torch.stack(batch, dim=0)
 
    out = model( batch )
    print("Input batch:\n", batch)
    print("\nOutput shape:", out.shape)
 
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
 
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)
 
    model.eval() # disable dropout
    out = generate( model=model, idx=encoded_tensor, max_new_tokens=6, context_size=cfg["context_length"] )
 
    print("Output:", out)
    print( tokenizer.decode(out.squeeze(0).tolist()) )
    
else: 
    print(f'GPT imported from local file "{__name__}.py"')
 