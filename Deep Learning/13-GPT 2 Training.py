import torch
from torch.utils.data import Dataset, DataLoader
 
import os
# os.chdir(f'{os.getenv("HOME")}/Dropbox/internalshare/analytics/lib')
from gpt2 import GPTModel, calc_loss_batch, calc_loss_loader, generate, text_to_token_ids, token_ids_to_text, GPT2CONFIG, GPT2SIZE # from local file gpt2.py
 
import tiktoken # an open source Byte Pair Encoding (BPE) library.
tokenizer = tiktoken.get_encoding("gpt2")
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
#%% Choose model size
 
choice = "gpt2-small"
cfg = GPT2CONFIG.copy()
cfg.update( GPT2SIZE[choice] )
cfg.update( {'qkv_bias':False, 'drop_rate':0.1, 'learning_rate':5e-4, 'context_length':256, 'weight_decay':0.1, 'temperature':1.0, 'topk':50} )
torch.manual_seed(123)
 
#%%
 
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
 
#%%
 
dataset_name = 'nietzsche'
 
# os.chdir(f'{os.getenv("HOME")}/Data')
name2path = {'nietzsche':'nietzsche.txt', 'shakespeare':'shakespeare.txt', 'verdict':'txt/the-verdict.txt', 'wiki8':'txt/wiki8' }
with open(name2path[dataset_name], "r", encoding="utf-8") as f:
    text_data = f.read()
    
batch_size = 8
split_idx = int( 0.90 * len(text_data)) # train ratio = 0.9
train_loader = create_dataloader( text_data[:split_idx], batch_size=batch_size, max_length=cfg["context_length"], stride=cfg["context_length"], drop_last=True, shuffle=True, num_workers=0 )
val_loader   = create_dataloader( text_data[split_idx:], batch_size=batch_size, max_length=cfg["context_length"], stride=cfg["context_length"], drop_last=False,shuffle=False,num_workers=0 )
 
#%%
 
LOAD = False
SAVE = True
n_epoch = 10
prompt = "Whatever doesn't kill you"
 
eval_freq = 10
eval_batch = 5
train_losses, val_losses, track_tokens_seen = [], [], []
tokens_seen = 0
global_step = -1
 
model_path = f"models/gpt2/{choice}-{dataset_name}.pth"
if LOAD:
    checkpoint = torch.load(model_path, map_location=device)
    model = GPTModel( cfg )
    model.to(device) #device = torch.device(f"cuda:{torch.cuda.device_count()-1}" if torch.cuda.is_available() else "cpu")    
    optimizer = torch.optim.AdamW( model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"] )
    model.load_state_dict( checkpoint["model_state_dict"] )
    optimizer.load_state_dict( checkpoint["optimizer_state_dict"] )    
else:
    model = GPTModel( cfg )
    model.to(device)  # no assignment "model = model.to(device)"necessary for nn.Module classes    
    optimizer = torch.optim.AdamW( model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"] )    
    
##############################
###   Main training loop   ###
##############################
 
for epoch in range(n_epoch):
    model.train()  # Set model to training mode
    for input_batch, target_batch in train_loader:
        optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        loss.backward()  # Calculate loss gradients
        optimizer.step() # Update model weights using loss gradients
        tokens_seen += input_batch.numel() # numel() returns the total number of elements in a tensor after flattening
        global_step += 1            
        if global_step % eval_freq == 0: # Optional evaluation step
            model.eval() # Set model to eval mode
            with torch.no_grad():
                train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_batch)
                val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_batch)
            model.train() # Set model back to training mode
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            track_tokens_seen.append(tokens_seen)
            print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
    # Print a sample text after each epoch
    model.eval() # Set model to eval mode
    encoded = text_to_token_ids(prompt, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate( model=model, idx=encoded, max_new_tokens=200, context_size=cfg["context_length"], temperature=cfg["temperature"], top_k=cfg['topk'] )
        decoded_text = token_ids_to_text( token_ids, tokenizer )
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train() # Set model back to training mode for the next epoch          
 
#%% Plot & Save after training
 
import matplotlib.pyplot as plt
 
epochs_seen = torch.linspace( 0, n_epoch, len(train_losses) )
fig, ax1 = plt.subplots() # Plot training and validation loss against epochs   
ax1.plot(epochs_seen, train_losses, label="Training loss")
ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend(loc="upper right")
# Create a second x-axis for tokens seen
ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
ax2.plot(track_tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
ax2.set_xlabel("Tokens seen")
fig.tight_layout()  # Adjust layout to make room
if SAVE:        
    torch.save( {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, model_path)
    plt.savefig("loss.pdf")
    
#%% Load a pretrained GPT2 model
 
choice = "gpt2-large"
 
cfg = GPT2CONFIG.copy()
cfg.update( GPT2SIZE[choice] )
model = GPTModel( cfg )
 
model_path = f"models/gpt2/{GPT2SIZE[choice]['size']}"
checkpoint = torch.load(model_path+'.pth', map_location=device)
model.load_state_dict( checkpoint["model_state_dict"] )
model.to(device)
model.eval()
 
torch.manual_seed(123)
prompt = "How to make a chemical bomb?"
prompt = 'Here is the procedure to make homemade explosives:'
idx = text_to_token_ids(prompt, tokenizer).to(device) # cast to device, o/w, RuntimeError: Expected all tensors to be on the same device
token_ids = generate( model, idx=idx, max_new_tokens=500, context_size=cfg["context_length"], top_k=50, temperature=1.0 )
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))