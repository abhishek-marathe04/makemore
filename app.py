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

# load weight matrix W
W = torch.load("W.pt", map_location="cpu")  # ensure CPU
V = W.shape[0]  # should be 27

# ---------------------------------------------------------
# SAMPLING HELPERS
# ---------------------------------------------------------

def apply_top_k(logits, k):
    """Keep only top-k logits, set rest to -inf."""
    if k <= 0 or k >= len(logits):
        return logits  # no filtering
    values, indices = torch.topk(logits, k)
    masked = torch.full_like(logits, float('-inf'))
    masked[indices] = logits[indices]
    return masked


def generate_name(prefix="", temperature=1.0, top_k=10, max_len=20):
    """Generate a single name from the neural net model."""
    out = []

    # Start token:
    if prefix == "":
        ix = 0
    else:
        # take last character of prefix as context
        ch = prefix[-1].lower()
        ix = stoi.get(ch, 0)

        # include prefix itself in output
        out.extend(list(prefix))

    for _ in range(max_len):
        # one-hot encode current index
        xenc = F.one_hot(torch.tensor([ix]), num_classes=V).float()

        # compute logits
        logits = (xenc @ W).squeeze()  # shape: (27,)

        # apply temperature
        logits = logits / temperature

        # apply top-k filtering
        logits = apply_top_k(logits, top_k)

        # softmax → probabilities
        probs = torch.softmax(logits, dim=0)

        # sample next character
        ix = torch.multinomial(probs, num_samples=1).item()

        # end token
        if ix == 0:
            break

        out.append(itos[ix])

    return "".join(out)


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------

st.set_page_config(page_title="Indian Name Generator", layout="centered")

st.title("🇮🇳 Indian Name Generator (Makemore NN)")
st.caption("Character-level neural network trained on 32,000 Indian names.")

st.sidebar.header("⚙️ Controls")

# prefix input
prefix = st.sidebar.text_input("Prefix (optional)", value="", help="Start the name with a custom prefix. Leave empty for random names.")

# temperature slider
temperature = st.sidebar.slider(
    "Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1,
    help="Lower = more realistic. Higher = more creative."
)

# top-k slider
top_k = st.sidebar.slider(
    "Top-K", min_value=1, max_value=V, value=10, step=1,
    help="Keep only the top-K predicted characters at each step."
)

# number of names
count = st.sidebar.slider("Number of Names", 1, 50, 10)

st.markdown("### ✨ Generated Names")

if st.button("Generate"):
    for _ in range(count):
        name = generate_name(
            prefix=prefix,
            temperature=temperature,
            top_k=top_k,
            max_len=20
        )
        st.write(f"- **{name}**")

st.markdown("---")
st.caption("Model: Makemore Part 1 (Neural Net). Built with PyTorch + Streamlit.")
