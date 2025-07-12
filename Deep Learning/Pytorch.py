#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import torch
import torch.nn as nn
 
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs available:", torch.cuda.device_count())
 
torch.manual_seed(123)
#%%
 
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
 
x_2 = inputs[1] # second input element
d_in = inputs.shape[1] # the input embedding size, d=3
d_out = 2 # the output embedding size, d=2
 
W_query = torch.nn.Parameter( torch.rand(d_in, d_out), requires_grad=False )
 
x_2 @ W_query
torch.matmul(x_2, W_query) # @ is the same as matmul()
torch.matmul(W_query, x_2) # RuntimeError: size mismatch
 
#%%
class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
 
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 30), torch.nn.ReLU(),
            torch.nn.Linear(30, 20), torch.nn.ReLU(),
            torch.nn.Linear(20, num_outputs) )
 
    def forward(self, x):
        logits = self.layers(x)
        return logits    
 
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
    
class ExampleDNN(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential( nn.Linear(layer_sizes[0], layer_sizes[1]), GELU() ),
            nn.Sequential( nn.Linear(layer_sizes[1], layer_sizes[2]), GELU() ),
            nn.Sequential( nn.Linear(layer_sizes[2], layer_sizes[3]), GELU() ),
            nn.Sequential( nn.Linear(layer_sizes[3], layer_sizes[4]), GELU() ),
            nn.Sequential( nn.Linear(layer_sizes[4], layer_sizes[5]), GELU() )
        ])
 
    def forward(self, x):
        for layer in self.layers:           
            layer_output = layer(x) # Compute the output of the current layer            
            if self.use_shortcut and x.shape == layer_output.shape: # Check if shortcut can be applied
                x = x + layer_output # Residual Learning
            else:
                x = layer_output
        return x
 
 
def print_gradients(model, x):   
    output = model(x)  # Forward pass
    target = torch.tensor([[0.]])  
    loss = nn.MSELoss()(output, target)
    loss.backward() # Backward pass to calculate the gradients
    for name, param in model.named_parameters():
        if 'weight' in name: # Print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
           
#%%            
layer_sizes = [3, 3, 3, 3, 3, 1]  
sample_input = torch.tensor([[1., 0., -1.]])
 
model_without_shortcut = ExampleDNN( layer_sizes, use_shortcut=False )
print_gradients(model_without_shortcut, sample_input)
 
model_with_shortcut = ExampleDNN( layer_sizes, use_shortcut=True )
print_gradients(model_with_shortcut, sample_input)
 
#%%
 
from torch.utils.data import Dataset, DataLoader
 
class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y
        
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y
 
    def __len__(self):
        return self.labels.shape[0]
 
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5] ])
y_train = torch.tensor([0, 0, 0, 1, 1])
X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6] ])
y_test = torch.tensor([0, 1])
 
train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)
 
I = iter(train_ds) # a Dataset object is an iterable 
I is train_ds # False: the iterable is not its own iterator, just like TF Dataset
next(I)
 
train_loader = DataLoader( dataset=train_ds, batch_size=2, shuffle=False, pin_memory=True, drop_last=True )
test_loader = DataLoader(  dataset=test_ds,  batch_size=2, shuffle=False, )
 
I = iter( train_loader)
I is train_loader # False
next(I)
 
#%%
 
import tiktoken # an open source Byte Pair Encoding (BPE) library.
tokenizer = tiktoken.get_encoding("gpt2")
# tiktoken uses <|endoftext|> as both end-of-sentence token and the padding token
text = ("I like traveling. <|endoftext|> Once I visited someunknownPlace.")
token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
token_ids # <|endoftext|> is assigned the largest token ID of a vocabulary of 50,257 tokens
tokenizer.decode(token_ids)
 
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []        
        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})
        # Use a sliding window to chunk txt into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[ i+1 : i+max_length+1 ]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
 
    def __len__(self):
        return len(self.input_ids)
 
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
 
def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):    
    tokenizer = tiktoken.get_encoding("gpt2") # Initialize the tokenizer    
    dataset = GPTDataset(txt, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
 
 
with open('txt/the-verdict.txt', "r", encoding="utf-8") as f:
    text_data = f.read()
 
# use the same number for max_length and stride to avoid overlapping which reduces overfitting
dataloader = create_dataloader( text_data, batch_size=8, max_length=4, stride=4, shuffle=False)
I = iter(dataloader)
next(I)