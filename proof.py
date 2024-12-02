import numpy as np
import torch
from torch.fft import fft, ifft
import matplotlib.pyplot as plt


def freq_init(shape, scale=1.0):
    """Initialize weights directly in frequency domain with proper conjugate symmetry"""
    n = shape[0]
    freqs = np.zeros(shape, dtype=np.complex64)
    # Only fill first half + middle frequency (nyquist)
    half_n = (n // 2) + 1
    freqs[:half_n] = (
        (np.random.randn(half_n, *shape[1:]) + 1j * np.random.randn(half_n, *shape[1:]))
        * scale
        / np.sqrt(n)
    )
    # Ensure conjugate symmetry for real signal
    freqs[half_n:] = np.conj(freqs[1 : n - half_n + 1][::-1])
    # Make DC component real
    freqs[0] = freqs[0].real
    if n % 2 == 0:  # Make nyquist frequency real if even length
        freqs[n // 2] = freqs[n // 2].real
    return torch.from_numpy(freqs)


def verify_equivalence(seq_len=32, embed_dim=8, batch_size=2, threshold=1e-4):
    print("\nInitializing test with:")
    print(f"Sequence length: {seq_len}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Batch size: {batch_size}\n")

    # Initialize input and weights in frequency domain
    x_f = freq_init((seq_len, batch_size, embed_dim))
    Wq_f = freq_init((seq_len, embed_dim, embed_dim))
    Wk_f = freq_init((seq_len, embed_dim, embed_dim))

    print("Computing spatial domain attention...")
    # Convert to spatial domain for traditional attention
    x = ifft(x_f, dim=0).real  # (seq_len, batch, embed)
    Wq = ifft(Wq_f, dim=0).real  # (seq_len, embed, embed)
    Wk = ifft(Wk_f, dim=0).real  # (seq_len, embed, embed)

    # Traditional attention in spatial domain
    Q = x @ Wq  # (seq_len, batch, embed)
    K = x @ Wk  # (seq_len, batch, embed)
    scores_spatial = Q @ K.transpose(1, 2)  # (seq_len, batch, batch)

    print("Computing frequency domain attention...")
    # Frequency domain computation
    # Compute Q = x @ Wq in frequency domain
    Q_f = (
        torch.sum(x_f[:, :, None, :] * Wq_f[:, None, :, :], dim=-1) / seq_len
    )  # (seq, batch, embed)

    # Compute K = x @ Wk in frequency domain
    K_f = (
        torch.sum(x_f[:, :, None, :] * Wk_f[:, None, :, :], dim=-1) / seq_len
    )  # (seq, batch, embed)

    # Compute Q @ K.T in frequency domain
    scores_freq = (
        torch.sum(Q_f[:, :, None, :] * torch.conj(K_f[:, None, :, :]), dim=-1) / seq_len
    )  # (seq, batch, batch)
    scores_freq = ifft(scores_freq, dim=0).real * seq_len  # Scale back after IFFT

    print("\nShape information:")
    print(f"Spatial scores shape: {scores_spatial.shape}")
    print(f"Frequency scores shape: {scores_freq.shape}")

    print("\nValue ranges:")
    print(
        f"Spatial scores range: [{scores_spatial.min():.6f}, {scores_spatial.max():.6f}]"
    )
    print(f"Frequency scores range: [{scores_freq.min():.6f}, {scores_freq.max():.6f}]")

    # Convert to numpy for comparison and plotting
    scores_spatial = scores_spatial.numpy()
    scores_freq = scores_freq.numpy()

    # Verify equivalence
    max_diff = np.max(np.abs(scores_spatial - scores_freq))
    rel_diff = max_diff / (np.abs(scores_spatial).max() + 1e-8)
    print(f"\nAbsolute maximum difference: {max_diff:.8f}")
    print(f"Relative maximum difference: {rel_diff:.8f}")

    # Plot comparison
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(scores_spatial[0])
    plt.title("Spatial Domain Scores")
    plt.colorbar()

    plt.subplot(132)
    plt.imshow(scores_freq[0])
    plt.title("Frequency Domain Scores")
    plt.colorbar()

    plt.subplot(133)
    plt.imshow(np.abs(scores_spatial[0] - scores_freq[0]))
    plt.title("Absolute Difference")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("attention_equivalence.png")
    plt.close()

    return rel_diff < threshold  # Use relative difference with threshold


if __name__ == "__main__":
    np.random.seed(42)
    is_equivalent = verify_equivalence(
        threshold=1e-4
    )  # Allow 0.01% relative difference
    print(f"\nAttention mechanisms equivalent within threshold: {is_equivalent}")
