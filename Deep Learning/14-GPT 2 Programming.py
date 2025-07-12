import torch
import random
import json

import os

# os.chdir(f'{os.getenv("HOME")}/Dropbox/internalshare/analytics/lib')
from gpt2 import GPTModel, generate, text_to_token_ids, token_ids_to_text, GPT2CONFIG, \
    GPT2SIZE  # from local file gpt2.py

os.chdir(f'{os.getenv("HOME")}/Data/')
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%

def format_input(x):
    instruction = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{x['instruction']}"
    input_text = f"\n\n### Input:\n{x['input']}" if x["input"] else ""
    return instruction + input_text


# %%

choice = "gpt2-large"

cfg = GPT2CONFIG.copy()
cfg.update(GPT2SIZE[choice])
model = GPTModel(cfg)
model_state_dict = torch.load(f"models/gpt2//{choice}-sft.pth", map_location=device)
model.load_state_dict(model_state_dict)

model.to(device)

# %% Supervised Fine-Tuning with Instructions

file_path = 'instruction/instruction1100.json'
with open(file_path, "r") as file:
    data = json.load(file)

# %% Generate responses

entry = random.choice(data)

entry = {'instruction': "What is the state capital of the following state",
         'input': 'Washington'}

entry = {'instruction': "Evaluate the performance of the following U.S. president",
         'input': 'Donald Trump'}

entry = {'instruction': "What is the square of 5", 'input': ''}

# %%
prompt = format_input(entry)
token_ids = generate(model, idx=text_to_token_ids(prompt, tokenizer).to(device), max_new_tokens=100,
                     context_size=cfg["context_length"], eos_id=50256)
generated_text = token_ids_to_text(token_ids, tokenizer)
response = generated_text[len(prompt):].replace("### Response:", "").strip()
print(f'{prompt}\n\n### Output:\n')
print(response)