# app.py
import streamlit as st
import torch
import torch.nn.functional as F
import json

W = torch.load("W.pt")


with open("itos.json", "r") as f:
    itos_dict = json.load(f)
    # convert keys (strings) to ints, sort, then create a list
    itos = [itos_dict[str(i)] for i in range(len(itos_dict))]

stoi = {ch:i for i,ch in enumerate(itos)}

# ------------------------------------------------------------
# Assumptions: You already have W, itos, stoi defined somewhere.
# You can import them or load them here.
# Example:
# from model import W, itos, stoi
# ------------------------------------------------------------

# --- Replace this block with your actual model loading -----
# Dummy placeholders (REMOVE THESE when using your real model)
# itos = ['.'] + list("abcdefghijklmnopqrstuvwxyz")
# stoi = {ch:i for i,ch in enumerate(itos)}
# W = torch.randn(27, 27) * 0.1   # load your trained W instead
# ------------------------------------------------------------

st.title("Random Indian Name Generator (Makemore Style)")
st.write("This demo generates random names using a character-level neural network trained on 32,000 names.")

# Number of names user wants
n = st.slider("How many names to generate?", 1, 50, 10)

generator = torch.Generator().manual_seed(2147483647)

def generate_name():
    out = []
    ix = 0  # start token index
    while True:
        # one-hot encode
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        
        # forward pass using your trained W
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)

        # sample next character
        ix = torch.multinomial(p, num_samples=1, generator=generator).item()
        out.append(itos[ix])

        if ix == 0:  # end token
            break

    return ''.join(out)

if st.button("Generate Names"):
    st.subheader("Generated Names:")
    for _ in range(n):
        st.write(generate_name())
