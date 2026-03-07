# app.py
import streamlit as st
import torch
import torch.nn.functional as F
import json
# ---------------------------------------------------------
# LOAD MODEL ARTIFACTS (W, itos)
# ---------------------------------------------------------

with open("itos.json", "r") as f:
    data = json.load(f)

# handle dict OR list format
if isinstance(data, list):
    itos = data
else:
    # convert dict {"0":"a", "1":"b", ...} → list
    itos = [data[str(i)] for i in range(len(data))]

stoi = {ch: i for i, ch in enumerate(itos)}
block_size = 3

# load weight matrix W
params = torch.load("model_part2.pt", map_location="cpu")  # ensure CPU
C = params["C"]
W1 = params["W1"]
b1 = params["b1"]
W2 = params["W2"]
b2 = params["b2"]

def apply_top_k(logits, k):
    """Keep only top-k logits, set rest to -inf."""
    if k <= 0 or k >= len(logits):
        return logits  # no filtering
    values, indices = torch.topk(logits, k)
    masked = torch.full_like(logits, float('-inf'))
    masked[indices] = logits[indices]
    return masked

def generate_name_part2(prefix="", temperature=1.0, top_k=10, max_len=20):
    """Generate a single name from the neural net model."""
    names = []

    # Start token:
    if prefix == "":
        ix = 0
    else:
        # take last character of prefix as context
        ch = prefix[-1].lower()
        ix = stoi.get(ch, 0)

        # include prefix itself in output
        names.extend(list(prefix))

    # for _ in range(max_len):
    #     # one-hot encode current index
    #     xenc = F.one_hot(torch.tensor([ix]), num_classes=V).float()

    #     # compute logits
    #     logits = (xenc @ W).squeeze()  # shape: (27,)

    #     # apply temperature
    #     logits = logits / temperature

    #     # apply top-k filtering
    #     logits = apply_top_k(logits, top_k)

    #     # softmax → probabilities
    #     probs = torch.softmax(logits, dim=0)

    #     # sample next character
    #     ix = torch.multinomial(probs, num_samples=1).item()

    #     # end token
    #     if ix == 0:
    #         break

    #     out.append(itos[ix])
    g = torch.Generator().manual_seed(2147483647) # for reproducability

    # for _ in range(20):

    out = []
    context = [0] * block_size # Initilise with all ...

    while True:
        emb = C[torch.tensor([context])] # (1, block_size, d)
        h = torch.tanh(emb.view(1, -1) @ W1 + b1) # (37,100)
        logits = h @ W2 + b2 # 37,27

        logits = apply_top_k(logits, top_k)
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break

    names.append(''.join(itos[i] for i in out))
    print(names)
    # print('Returning ...')
    return "".join(names)