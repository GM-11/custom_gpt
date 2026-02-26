# %%
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from custom_dataset import CustomDataset

# %%
sentences = open("transcripts.txt", "r").read().split("\n")

# %%
encoder = tiktoken.encoding_for_model("gpt-2")
data = torch.tensor(encoder.encode("".join(sentences)), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:100])

# %%
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# %%
seq_len = 8
inputs = []
targets = []

# %%
encoded_sentences = []
for s in sentences:
    encoded_sentences.append(encoder.encode(s))

# %%
for ids in encoded_sentences:
    for i in range(len(ids) - seq_len):
        window = ids[i : i + seq_len]
        target = ids[i + seq_len]

        inputs.append(list(window))
        targets.append(target)

# %%
for i, t in zip(inputs, targets):
    print(i, encoder.decode(i), " -> ", encoder.decode([t]))

# %%
dataset = CustomDataset(inputs, targets)

loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# %%
sent = sentences[0]
tokens = encoder.encode(sent)
print(tokens)
# %%

print(len(data))
print(max(tokens))
print(encoder.n_vocab)

# %%
embedding_dim = 64
torch.manual_seed(42)
embed = nn.Embedding(encoder.n_vocab, embedding_dim)
x = embed(torch.tensor(tokens).unsqueeze(0))
x
from manual_self_attention import ManualSelfAttention

attention_layer = ManualSelfAttention(embedding_dim)

out, attn = attention_layer(x)
print(out[0].shape)
print(attn[0].shape)
# %%
for i, w in enumerate(tokens):
    row = ["{:.2f}".format(a) for a in attn[0, i].detach().cpu().numpy()]
    print(f"{encoder.decode([w]):>8} attends to -> {row}")

# %%
from self_attenstion_with_position import SelfAttnWithPositionalEmbedding

model = SelfAttnWithPositionalEmbedding(
    vocab_size=encoder.n_vocab, seq_len=seq_len, embedding_dim=embedding_dim
)

model
# %%
model.eval()
x_example = torch.tensor([inputs[0]], dtype=torch.long)
print(x_example.shape)
with torch.no_grad():
    logits, attn_weights = model(x_example)

# %%
from model_training import train_model

train_model(
    model=model,
    loader=loader,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
)

# %%
input_ids = inputs[22]
human_readable_tokens = [encoder.decode([id]) for id in input_ids]
model.eval()
x_example_eval = torch.tensor([input_ids], dtype=torch.long)

with torch.no_grad():
    logits, attn_weights_eval = model(x_example_eval)

from utils import plot_attention

plot_attention(attn_weights_eval, human_readable_tokens)


# %%
def generate_next_words(input_str: str, max_tokens: int):
    model.eval()

    input_tokenized = encoder.encode(input_str)

    for _ in range(max_tokens):
        input_ids = torch.tensor([input_tokenized], dtype=torch.long)

        with torch.no_grad():
            logits, _ = model(input_ids)
            next_id = logits.argmax(dim=-1).item()

        next_word = encoder.decode([next_id])
        print(next_word)
        input_tokenized.append(next_id)

    return encoder.decode(input_tokenized)
