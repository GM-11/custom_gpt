import matplotlib.pyplot as plt


def plot_attention(attn_weights, tokens, title="Self-Attention Map"):
    """
    Plots a self-attention map for a single input sequence.
    attn_weights: tensor of shape [batch, seq_len, seq_len], typically from your model
    tokens: list of token strings (for axis labels)
    title: plot title string
    """
    # Take attention weights for the first sample in the batch and move to CPU
    aw = attn_weights[0].detach().cpu().numpy()
    plt.figure(figsize=(1.2 * len(tokens), 5))  # Adjust figure size by sequence length
    # Show the attention matrix as an image (color = strength of attention)
    plt.imshow(aw, cmap="Blues")
    # Label x- and y-axes with token words
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    # Add a colorbar to show attention strength scale
    plt.colorbar()
    # Display the plot title
    plt.title(title)
    # Neatly fit everything in the figure area
    plt.tight_layout()
    plt.show()
